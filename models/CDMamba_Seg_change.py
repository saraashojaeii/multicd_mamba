from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from models.mamba_customer import ConvMamba


# ---------------------------------------------------------------------------
# Primitive building blocks
# ---------------------------------------------------------------------------

class ContextBlock2D(nn.Module):
    """
    Bottleneck context module: fuses local dilated features with a global
    average-pool branch to capture both fine-grained structure and scene-level context.

    Input / output shape: [B, C, H, W]
    """
    def __init__(self, ch, norm_layer, act):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, dilation=1, bias=False),
            norm_layer(ch), act,
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False),
            norm_layer(ch), act,
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1, bias=False), act,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            norm_layer(ch), act,
        )

    def forward(self, x):
        loc  = self.local(x)                                                         # [B, C, H, W]
        glob = F.interpolate(self.global_branch(x), size=x.shape[2:],
                             mode="bilinear", align_corners=False)                   # [B, C, H, W]
        return self.fuse(torch.cat([loc, glob], dim=1))


class ConvPosEnc(nn.Module):
    """Depthwise-conv positional encoding: adds a locally-smoothed bias to x."""
    def __init__(self, dim, k=3):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=True)

    def forward(self, x):
        return x + self.proj(x)


class ModifiedSRCMLayer(nn.Module):
    """
    Core Mamba token-mixer for 2-D feature maps.

    Pipeline:
      1. Depthwise-conv positional encoding
      2. Flatten H×W → token sequence, add learned 2-D position embeddings
      3. Grouped bi-directional Mamba (each of G groups processes C/G channels)
      4. Gated residual: sigmoid(gate) * mamba_out + (1-gate) * input
      5. Linear projection to output_dim

    Input: [B, input_dim, H, W]   Output: [B, output_dim, H, W]
    """
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, groups=4):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.groups     = groups
        self.norm       = nn.LayerNorm(input_dim)
        self.mambas     = nn.ModuleList([
            ConvMamba(d_model=input_dim // groups, d_state=d_state, d_conv=d_conv,
                      expand=expand, bimamba_type="v2")
            for _ in range(groups)
        ])
        self.gate_proj = nn.Linear(input_dim, input_dim)
        self.pos_enc   = ConvPosEnc(input_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4096, input_dim))  # max 64×64 tokens
        self.proj      = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pos_enc(x)
        x = x.reshape(B, C, -1).transpose(1, 2)                          # [B, HW, C]

        pos = F.interpolate(
            self.pos_embed.transpose(1, 2).reshape(
                1, self.input_dim, int(self.pos_embed.shape[1] ** 0.5), -1),
            size=(H, W), mode='bilinear', align_corners=False,
        ).reshape(1, self.input_dim, -1).transpose(1, 2)                  # [1, HW, C]
        x = x + pos[:, :x.shape[1], :]

        x_norm  = self.norm(x)
        chunks  = x_norm.chunk(self.groups, dim=-1)
        x_mamba = torch.cat([m(c) for m, c in zip(self.mambas, chunks)], dim=-1)

        gate  = torch.sigmoid(self.gate_proj(x_norm))
        x_out = gate * x_mamba + (1 - gate) * x
        return self.proj(x_out).transpose(1, 2).reshape(B, self.output_dim, H, W)


def get_srcm_layer(spatial_dims, in_channels, out_channels, stride=1, conv_mode="deepwise"):
    """Build a ModifiedSRCMLayer, optionally followed by MaxPool2d for stride-2 downsampling."""
    layer = ModifiedSRCMLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1 and spatial_dims == 2:
        return nn.Sequential(layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return layer


class SRCMBlock(nn.Module):
    """
    Residual SRCM block (used in both encoder and decoder).

    Pre-norm → Act → SRCM → Pre-norm → Act → SRCM → Dropout → SE → residual

    The learnable residual scale allows the block to adaptively suppress or
    amplify its own contribution.

    Input / output: [B, C, H, W]
    """
    def __init__(self, spatial_dims, in_channels, norm,
                 act=("RELU", {"inplace": True}), kernel_size=3, conv_mode="deepwise"):
        super().__init__()
        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        self.norm1     = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2     = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act       = get_act_layer(act)
        self.conv1     = get_srcm_layer(spatial_dims, in_channels, in_channels, conv_mode=conv_mode)
        self.conv2     = get_srcm_layer(spatial_dims, in_channels, in_channels, conv_mode=conv_mode)
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        self.drop      = nn.Dropout2d(p=0.1)
        self.se        = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        identity = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        x = self.drop(x) * self.se(x)
        return identity + self.res_scale * x


# ---------------------------------------------------------------------------
# Encoder and Decoder
# ---------------------------------------------------------------------------

class SRCMEncoder(nn.Module):
    """
    Hierarchical feature encoder.

    Stage i produces features at 1/2^max(0,i) of the input resolution with
    init_filters * 2^i channels. Stage 0 has no downsampling; stages 1+ start
    with a stride-2 SRCM layer followed by n SRCMBlocks.

    forward(x) → (latent, all_skips)
        latent    : deepest feature map  [B, C_deep, H/8, W/8]
        all_skips : [scale_0, ..., scale_N]  ordered shallow → deep
                    (latent == all_skips[-1])
    """
    def __init__(self, spatial_dims, in_channels, init_filters, blocks_down,
                 norm, act, conv_mode, dropout_prob=None):
        super().__init__()
        self.dropout_prob = dropout_prob
        self.conv_init    = get_conv_layer(spatial_dims, in_channels, init_filters)
        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

        stages = []
        for i, n_blocks in enumerate(blocks_down):
            ch         = init_filters * 2 ** i
            downsample = (
                get_srcm_layer(spatial_dims, ch // 2, ch, stride=2, conv_mode=conv_mode)
                if i > 0 else nn.Identity()
            )
            stages.append(nn.Sequential(
                downsample,
                *[SRCMBlock(spatial_dims, ch, norm=norm, act=act, conv_mode=conv_mode)
                  for _ in range(n_blocks)],
            ))
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.conv_init(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return x, skips   # latent == skips[-1]


class SRCMDecoder(nn.Module):
    """
    Hierarchical feature decoder.

    At each step: upsample → spatially align if needed → concat skip → refine.

    forward(latent, skips_deep_to_shallow) → feature map at full encoder resolution

    skips_deep_to_shallow: encoder skips ordered deep → shallow, excluding the
                           bottleneck (deepest) scale which is the latent itself.
    """
    def __init__(self, spatial_dims, init_filters, blocks_up, norm, act,
                 up_conv_mode, upsample_mode):
        super().__init__()
        self.spatial_dims = spatial_dims
        n_up = len(blocks_up)

        up_samples, layers = [], []
        for i in range(n_up):
            in_ch  = init_filters * 2 ** (n_up - i)   # e.g. 128 → 64 → 32
            out_ch = in_ch // 2                        # e.g.  64 → 32 → 16
            up_samples.append(nn.Sequential(
                get_conv_layer(spatial_dims, in_ch, out_ch, kernel_size=1),
                get_upsample_layer(spatial_dims, out_ch, upsample_mode=upsample_mode),
            ))
            layers.append(nn.Sequential(
                get_conv_layer(spatial_dims, out_ch * 2, out_ch, kernel_size=3, stride=1),
                SRCMBlock(spatial_dims, out_ch, norm=norm, act=act, conv_mode=up_conv_mode),
            ))
        self.up_samples = nn.ModuleList(up_samples)
        self.layers     = nn.ModuleList(layers)

    def forward(self, x, skips):
        """
        x    : [B, C_deep, H_deep, W_deep]
        skips: encoder skip features ordered deep → shallow, len == n_up
        """
        interp = "bilinear" if self.spatial_dims == 2 else "trilinear"
        for up, layer, skip in zip(self.up_samples, self.layers, skips):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode=interp, align_corners=False)
            x = layer(torch.cat([x, skip], dim=1))
        return x


# ---------------------------------------------------------------------------
# Cross-temporal fusion modules
# ---------------------------------------------------------------------------

class CrossTemporalFusion(nn.Module):
    """
    Multi-cue differencing fusion at one encoder scale.

    Concatenates [F1, F2, |F2-F1|, F2-F1] (4C channels) and reduces back to C
    via a 1×1 conv (channel reduction) followed by a 3×3 conv (spatial refinement).

    Input : two feature maps F1, F2  each [B, C, H, W]
    Output: fused change features         [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, f1, f2):
        diff = f2 - f1
        return self.fuse(torch.cat([f1, f2, diff.abs(), diff], dim=1))


class JointFeatureLearning(nn.Module):
    """
    Bidirectional coupling between semantic (T1/T2) and change features.

    Step 1 — change → semantic:
        A sigmoid gate derived from change features multiplicatively modulates
        both T1 and T2 semantic features.

    Step 2 — semantic → change:
        The averaged (and already modulated) semantic features provide additive
        context that is refined and added back to the change features.

    All inputs / outputs: [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        self.change_to_sem_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.Sigmoid(),
        )
        self.sem_to_change_ctx = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.refine_change = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, f1, f2, change_feat):
        # Change gates semantic (multiplicative)
        gate   = self.change_to_sem_gate(change_feat)
        f1_enh = f1 * (1.0 + gate)
        f2_enh = f2 * (1.0 + gate)
        # Semantic enriches change (additive)
        sem_avg    = (f1_enh + f2_enh) / 2.0
        ctx        = self.sem_to_change_ctx(torch.cat([sem_avg, change_feat], dim=1))
        change_enh = self.refine_change(change_feat + ctx)
        return f1_enh, f2_enh, change_enh


class MultiScaleInteractionBlock(nn.Module):
    """
    Channel-attention interaction between semantic and change features.

    Avoids O(HW²) spatial attention by squeezing to per-channel descriptors first.
    Each path modulates the other via a squeeze-and-excitation network, then a
    depthwise-separable conv refines the result spatially.

    All inputs / outputs: [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        def _se():
            return nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1), nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid(),
            )
        def _dw_refine():
            return nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, channels, 1, bias=False), nn.ReLU(inplace=True),
            )
        # change features produce a gate for semantic channels, and vice-versa
        self.chg_to_sem_gate = _se()
        self.sem_to_chg_gate = _se()
        self.norm_sem        = nn.BatchNorm2d(channels)
        self.norm_chg        = nn.BatchNorm2d(channels)
        self.refine_sem      = _dw_refine()
        self.refine_chg      = _dw_refine()

    def forward(self, sem_t1, sem_t2, change_feat):
        sem_avg = (sem_t1 + sem_t2) / 2.0

        sem_modulated  = sem_avg     * self.chg_to_sem_gate(change_feat)   # change gates semantic
        chg_modulated  = change_feat * self.sem_to_chg_gate(sem_avg)       # semantic gates change

        sem_out = self.refine_sem(self.norm_sem(sem_avg     + sem_modulated))
        chg_out = self.refine_chg(self.norm_chg(change_feat + chg_modulated))

        return sem_t1 + sem_out, sem_t2 + sem_out, change_feat + chg_out


class SemanticChangeInteractionBlock(nn.Module):
    """
    Bottleneck-level cross-attention (or Mamba) between semantic and change tokens.

    Flattens the spatial dims into a token sequence, runs cross-attention in both
    directions (semantic ↔ change), applies a small FFN, then reshapes back.
    Used only at the deepest (smallest H×W) scale to keep compute tractable.

    All inputs / outputs: [B, C, H, W]
    """
    def __init__(self, channels, num_heads=4, use_mamba=False):
        super().__init__()
        self.channels  = channels
        self.use_mamba = use_mamba

        self.norm_sem = nn.LayerNorm(channels)
        self.norm_chg = nn.LayerNorm(channels)

        if use_mamba:
            # Lightweight Mamba-style cross-mixing (alternative to attention)
            self.chg_to_sem = ConvMamba(d_model=channels, d_state=16, d_conv=4,
                                        expand=2, bimamba_type="v2")
            self.sem_to_chg = ConvMamba(d_model=channels, d_state=16, d_conv=4,
                                        expand=2, bimamba_type="v2")
        else:
            # Cross-attention: semantic queries change context, and vice-versa
            self.sem_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
            self.chg_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.ffn_sem      = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels),
        )
        self.ffn_chg      = nn.Sequential(
            nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels),
        )
        self.norm_ffn_sem = nn.LayerNorm(channels)
        self.norm_ffn_chg = nn.LayerNorm(channels)

    def forward(self, sem_t1, sem_t2, change_fused):
        B, C, H, W = sem_t1.shape

        # [B, C, H, W] → [B, HW, C]
        sem_avg = (sem_t1.flatten(2).transpose(1, 2) +
                   sem_t2.flatten(2).transpose(1, 2)) / 2.0
        chg     = change_fused.flatten(2).transpose(1, 2)

        sem_n = self.norm_sem(sem_avg)
        chg_n = self.norm_chg(chg)

        if self.use_mamba:
            def _2d(tok): return tok.transpose(1, 2).reshape(B, C, H, W)
            def _tok(feat): return feat.flatten(2).transpose(1, 2)
            sem_enh = sem_n + _tok(self.chg_to_sem(_2d(chg_n)))
            chg_enh = chg_n + _tok(self.sem_to_chg(_2d(sem_n)))
        else:
            sem_enh, _ = self.sem_attn(query=sem_n, key=chg_n, value=chg_n)
            sem_enh     = sem_avg + sem_enh                  # residual
            chg_enh, _ = self.chg_attn(query=chg_n, key=sem_n, value=sem_n)
            chg_enh     = chg + chg_enh                      # residual

        sem_enh = sem_enh + self.ffn_sem(self.norm_ffn_sem(sem_enh))
        chg_enh = chg_enh + self.ffn_chg(self.norm_ffn_chg(chg_enh))

        # [B, HW, C] → [B, C, H, W]
        sem_out = sem_enh.transpose(1, 2).reshape(B, C, H, W)
        chg_out = chg_enh.transpose(1, 2).reshape(B, C, H, W)

        return sem_t1 + sem_out, sem_t2 + sem_out, change_fused + chg_out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class CDMamba_seg_cd(nn.Module):
    """
    Joint Segmentation & Change Detection model based on Mamba.

    Architecture overview (change-detection mode):

        ┌──────────────────────────────────────────────────────┐
        │  Shared encoder  (SRCMEncoder)                       │
        │    T1 and T2 images encoded with the same weights    │
        │                                                       │
        │  Bottleneck context  (ContextBlock2D)                │
        │    Applied independently to each temporal branch     │
        │                                                       │
        │  Per-scale fusion  (×N encoder stages)               │
        │    CrossTemporalFusion  → concat + diff features     │
        │    JointFeatureLearning → bidirectional sem↔change   │
        │    MultiScaleInteraction→ channel-attention coupling  │
        │                                                       │
        │  Bottleneck fusion  (same three steps as above)      │
        │    + SemanticChangeInteraction (cross-attention)      │
        │                                                       │
        │  Three independent decoders  (SRCMDecoder)           │
        │    decoder_t1    → T1 segmentation                   │
        │    decoder_t2    → T2 segmentation                   │
        │    decoder_change→ binary change map                 │
        │                                                       │
        │  Change-guided gating  (optional)                    │
        │    change_prob modulates seg decoder features         │
        │                                                       │
        │  Prediction heads                                     │
        │    seg_head_t1, seg_head_t2 → [B, num_classes, H, W] │
        │    change_head              → [B, 1, H, W]           │
        └──────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        init_filters: int = 16,
        in_channels: int = 1,
        num_classes: int = 7,
        use_change_head: bool = True,
        conv_mode: str = "deepwise",
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        up_conv_mode: str = "deepwise",
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        use_change_gating: bool = True,
        change_gate_alpha: float = 1.0,
        change_gate_beta: float = 0.2,
        change_gate_mode: str = "additive",
        use_interaction_block: bool = True,
        interaction_num_heads: int = 4,
        interaction_use_mamba: bool = False,
    ):
        super().__init__()

        # ---- validate and resolve deprecated norm_name ----
        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(
                    f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead."
                )
            norm = ("group", {"num_groups": num_groups})

        # ---- hyper-parameters needed at forward time ----
        self.spatial_dims        = spatial_dims
        self.num_classes         = num_classes
        self.use_change_head     = use_change_head
        self.use_change_gating   = use_change_gating
        self.change_gate_alpha   = change_gate_alpha
        self.change_gate_beta    = change_gate_beta
        self.change_gate_mode    = change_gate_mode
        self.use_interaction_block = use_interaction_block
        # kept for inspection / checkpointing
        self.init_filters        = init_filters
        self.norm                = norm
        self.act                 = act
        self.upsample_mode       = UpsampleMode(upsample_mode)

        act_mod       = get_act_layer(act)
        bottleneck_ch = init_filters * (2 ** (len(blocks_down) - 1))
        n_scales      = len(blocks_down)

        # ---------------------------------------------------------------
        # Shared encoder  (T1 and T2 share the same weights)
        # ---------------------------------------------------------------
        self.encoder = SRCMEncoder(
            spatial_dims, in_channels, init_filters, blocks_down,
            norm, act, conv_mode, dropout_prob,
        )

        # ---------------------------------------------------------------
        # Bottleneck context  (applied independently to each branch)
        # ---------------------------------------------------------------
        if spatial_dims == 2:
            norm_fn = lambda c: get_norm_layer(name=norm, spatial_dims=2, channels=c)
            self.bottleneck = ContextBlock2D(bottleneck_ch, norm_fn, act_mod)
        else:
            self.bottleneck = nn.Identity()

        # ---------------------------------------------------------------
        # Per-scale cross-temporal fusion
        # ---------------------------------------------------------------
        self.fuse_scales = nn.ModuleList([
            CrossTemporalFusion(init_filters * 2 ** i) for i in range(n_scales)
        ])
        self.fuse_bottleneck = CrossTemporalFusion(bottleneck_ch)

        # ---------------------------------------------------------------
        # Per-scale joint learning  (bidirectional sem ↔ change coupling)
        # ---------------------------------------------------------------
        self.joint_learning_scales = nn.ModuleList([
            JointFeatureLearning(init_filters * 2 ** i) for i in range(n_scales)
        ])
        self.joint_learning_bottleneck = JointFeatureLearning(bottleneck_ch)

        # ---------------------------------------------------------------
        # Per-scale channel-attention interaction  (optional)
        # ---------------------------------------------------------------
        if use_interaction_block:
            self.interaction_blocks = nn.ModuleList([
                MultiScaleInteractionBlock(init_filters * 2 ** i) for i in range(n_scales)
            ])
            self.interaction_bottleneck = SemanticChangeInteractionBlock(
                channels=bottleneck_ch,
                num_heads=interaction_num_heads,
                use_mamba=interaction_use_mamba,
            )

        # ---------------------------------------------------------------
        # Skip-connection alignment  (1×1 conv per scale, per path)
        # ---------------------------------------------------------------
        self.skip_align_t1 = nn.ModuleList([
            nn.Conv2d(init_filters * 2 ** i, init_filters * 2 ** i, 1, bias=False)
            for i in range(n_scales)
        ])
        self.skip_align_t2 = nn.ModuleList([
            nn.Conv2d(init_filters * 2 ** i, init_filters * 2 ** i, 1, bias=False)
            for i in range(n_scales)
        ])
        self.skip_align_change = nn.ModuleList([
            nn.Conv2d(init_filters * 2 ** i, init_filters * 2 ** i, 1, bias=False)
            for i in range(n_scales)
        ])

        # ---------------------------------------------------------------
        # Decoders  (one per output path + one for single-image mode)
        # ---------------------------------------------------------------
        def _make_decoder():
            return SRCMDecoder(
                spatial_dims, init_filters, blocks_up,
                norm, act, up_conv_mode, self.upsample_mode,
            )

        self.decoder        = _make_decoder()   # single-image mode
        self.decoder_t1     = _make_decoder()
        self.decoder_t2     = _make_decoder()
        self.decoder_change = _make_decoder()

        # ---------------------------------------------------------------
        # Prediction heads
        # ---------------------------------------------------------------
        def _seg_head():
            return nn.Sequential(
                get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_filters),
                act_mod,
                get_conv_layer(spatial_dims, init_filters, num_classes, kernel_size=1, bias=True),
            )

        self.seg_head_t1 = _seg_head()
        self.seg_head_t2 = _seg_head()

        if use_change_head:
            # 1-channel logit; trained with BCE+Dice, predicted via sigmoid.
            # Single channel avoids the softmax-vs-sigmoid mismatch that arises
            # when a 2-channel head is trained with binary targets.
            self.change_head = nn.Sequential(
                get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_filters),
                act_mod,
                get_conv_layer(spatial_dims, init_filters, 1, kernel_size=1, bias=True),
            )

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _fuse_encoder_scales(
        self,
        skips1: list[torch.Tensor],
        skips2: list[torch.Tensor],
    ) -> tuple[list, list, list]:
        """
        At each encoder scale: CrossTemporalFusion → JointFeatureLearning →
        (optional) MultiScaleInteraction.

        Returns three parallel lists (shallow → deep):
            skips1_enh, skips2_enh, skips_fused
        """
        skips1_enh, skips2_enh, skips_fused = [], [], []
        for i, (f1, f2) in enumerate(zip(skips1, skips2)):
            fused       = self.fuse_scales[i](f1, f2)
            f1, f2, fused = self.joint_learning_scales[i](f1, f2, fused)
            if self.use_interaction_block:
                f1, f2, fused = self.interaction_blocks[i](f1, f2, fused)
            skips1_enh.append(f1)
            skips2_enh.append(f2)
            skips_fused.append(fused)
        return skips1_enh, skips2_enh, skips_fused

    def _fuse_bottleneck(
        self,
        lat1: torch.Tensor,
        lat2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fuse the two bottleneck feature maps with the same three-step pipeline
        used at encoder scales, plus the more powerful SemanticChangeInteraction.

        Returns enhanced (lat1, lat2, fused_latent).
        """
        fused          = self.fuse_bottleneck(lat1, lat2)
        lat1, lat2, fused = self.joint_learning_bottleneck(lat1, lat2, fused)
        if self.use_interaction_block:
            lat1, lat2, fused = self.interaction_bottleneck(lat1, lat2, fused)
        return lat1, lat2, fused

    def _aligned_decoder_skips(
        self,
        skips: list[torch.Tensor],
        align_modules: nn.ModuleList,
    ) -> list[torch.Tensor]:
        """
        Apply per-scale 1×1 alignment convolutions and return the result in
        deep → shallow order, excluding the deepest scale (which is the
        bottleneck / latent and is passed separately to the decoder).
        """
        aligned = [align(skip) for align, skip in zip(align_modules, skips)]
        # Drop the deepest scale (index -1) and reverse so decoder sees deep→shallow
        return list(reversed(aligned[:-1]))

    def _apply_change_gate(
        self,
        change_logits: torch.Tensor,
        seg1: torch.Tensor,
        seg2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Modulate segmentation decoder features with the predicted change probability.

        Additive mode:       feat * (1 + alpha * p)          amplifies changed regions
        Multiplicative mode: feat * (beta + (1-beta) * p)    suppresses unchanged regions
        """
        p = torch.sigmoid(change_logits)   # [B, 1, H, W]
        if p.shape[2:] != seg1.shape[2:]:
            interp = "bilinear" if self.spatial_dims == 2 else "trilinear"
            p = F.interpolate(p, size=seg1.shape[2:], mode=interp, align_corners=False)

        if self.change_gate_mode == "additive":
            seg1 = seg1 * (1.0 + self.change_gate_alpha * p)
            seg2 = seg2 * (1.0 + self.change_gate_alpha * p)
        else:   # multiplicative
            gate = self.change_gate_beta + (1.0 - self.change_gate_beta) * p
            seg1, seg2 = seg1 * gate, seg2 * gate

        return seg1, seg2

    # -------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None):
        """
        Single-image mode  (x2 is None):
            Returns: seg_logits_t1  [B, num_classes, H, W]

        Change-detection mode  (x2 provided, use_change_head=True):
            Returns: (seg_logits_t1, seg_logits_t2, change_logits)
                seg_logits_t1/t2 : [B, num_classes, H, W]
                change_logits    : [B, 1, H, W]

        Change-detection mode  (x2 provided, use_change_head=False):
            Returns: (seg_logits_t1, seg_logits_t2)
        """
        # ------------------------------------------------------------------ #
        #  Single-image segmentation mode                                     #
        # ------------------------------------------------------------------ #
        if x2 is None:
            latent, skips = self.encoder(x1)
            latent        = self.bottleneck(latent)
            dec_skips     = list(reversed(skips[:-1]))          # deep → shallow, no bottleneck
            dec           = self.decoder(latent, dec_skips)
            return self.seg_head_t1(dec)

        # ------------------------------------------------------------------ #
        #  Change-detection mode                                              #
        # ------------------------------------------------------------------ #

        # 1. Encode both images with the shared encoder
        lat1, skips1 = self.encoder(x1)
        lat2, skips2 = self.encoder(x2)

        # 2. Bottleneck context applied independently to each branch
        lat1 = self.bottleneck(lat1)
        lat2 = self.bottleneck(lat2)

        # 3. Per-scale fusion: CrossTemporalFusion → JointLearning → Interaction
        skips1_enh, skips2_enh, skips_fused = self._fuse_encoder_scales(skips1, skips2)

        # 4. Bottleneck-level fusion + interaction
        lat1, lat2, lat_fused = self._fuse_bottleneck(lat1, lat2)

        # 5. Align skip connections for each decoder path (deep→shallow, no bottleneck)
        t1_skips  = self._aligned_decoder_skips(skips1_enh,  self.skip_align_t1)
        t2_skips  = self._aligned_decoder_skips(skips2_enh,  self.skip_align_t2)
        chg_skips = self._aligned_decoder_skips(skips_fused, self.skip_align_change)

        # 6. Decode each path independently
        seg1_feats = self.decoder_t1(lat1,      t1_skips)
        seg2_feats = self.decoder_t2(lat2,      t2_skips)

        if self.use_change_head:
            chg_feats     = self.decoder_change(lat_fused, chg_skips)
            change_logits = self.change_head(chg_feats)

            # 7. (Optional) change-guided gating of seg decoder features
            if self.use_change_gating:
                seg1_feats, seg2_feats = self._apply_change_gate(
                    change_logits, seg1_feats, seg2_feats
                )

            return self.seg_head_t1(seg1_feats), self.seg_head_t2(seg2_feats), change_logits

        return self.seg_head_t1(seg1_feats), self.seg_head_t2(seg2_feats)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CDMamba_seg_cd(
        spatial_dims=2,
        in_channels=3,
        num_classes=6,
        use_change_head=True,
        init_filters=16,
        up_conv_mode="deepwise",
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    ).to(device)

    # Single-image segmentation mode
    seg = model(torch.randn(1, 3, 256, 256).to(device))
    print("Single-image mode output shape:", seg.shape)

    # Change-detection mode
    seg_t1, seg_t2, change = model(
        torch.randn(1, 3, 256, 256).to(device),
        torch.randn(1, 3, 256, 256).to(device),
    )
    print("Change-detection mode:")
    print(f"  Segmentation T1: {seg_t1.shape}")
    print(f"  Segmentation T2: {seg_t2.shape}")
    print(f"  Change map:      {change.shape}")
