from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from models.mamba_customer import ConvMamba

class ContextBlock2D(nn.Module):
    def __init__(self, ch, norm_layer, act):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, dilation=1, bias=False),
            norm_layer(ch),
            act,
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2, bias=False),
            norm_layer(ch),
            act,
        )
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch, 1, bias=False),
            act,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            norm_layer(ch),
            act,
        )

    def forward(self, x):
        loc = self.local(x)                            # [B, C, H, W]
        glob = self.global_branch(x)                   # [B, C, 1, 1]
        glob = F.interpolate(glob, size=x.shape[2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([loc, glob], dim=1))

class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        padding = k//2
        self.proj = nn.Conv2d(dim, dim, kernel_size=k, padding=padding, groups=dim, bias=True)
    def forward(self, x):  # x: [B,C,H,W]
        return x + self.proj(x)

class CrossTemporalFusion(nn.Module):
    """
    Multi-scale feature differencing fusion:
    concat(F1, F2, |F2-F1|, F2-F1) -> 1x1 conv reduce -> 3x3 conv refine.
    Output channels match input feature channels (C).
    """
    def __init__(self, channels: int):
        super().__init__()
        in_ch = channels * 4  # concat 4 feature maps
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Fuse T1 and T2 features at a given scale."""
        diff = f2 - f1
        adiff = torch.abs(diff)
        x = torch.cat([f1, f2, adiff, diff], dim=1)
        return self.fuse(x)

class SemanticChangeInteractionBlock(nn.Module):
    """
    Lightweight semantic-change interaction at bottleneck.
    Uses cross-attention between semantic tokens (T1, T2) and change tokens (fused).
    Minimal compute: single attention block with efficient implementation.
    """
    def __init__(self, channels: int, num_heads: int = 4, use_mamba: bool = False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_mamba = use_mamba
        
        # Layer norms
        self.norm_sem = nn.LayerNorm(channels)
        self.norm_chg = nn.LayerNorm(channels)
        
        if use_mamba:
            # Mamba-style mixing (lightweight alternative to attention)
            self.sem_to_chg = ConvMamba(d_model=channels, d_state=16, d_conv=4, expand=2, bimamba_type="v2")
            self.chg_to_sem = ConvMamba(d_model=channels, d_state=16, d_conv=4, expand=2, bimamba_type="v2")
        else:
            # Cross-attention: semantic attends to change, change attends to semantic
            self.sem_to_chg_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
            self.chg_to_sem_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
        # FFN for refinement
        self.ffn_sem = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.ffn_chg = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        
        self.norm_ffn_sem = nn.LayerNorm(channels)
        self.norm_ffn_chg = nn.LayerNorm(channels)
    
    def forward(self, sem_t1: torch.Tensor, sem_t2: torch.Tensor, change_fused: torch.Tensor):
        """
        Args:
            sem_t1: [B, C, H, W] - T1 semantic features
            sem_t2: [B, C, H, W] - T2 semantic features
            change_fused: [B, C, H, W] - fused change features
        Returns:
            Enhanced (sem_t1, sem_t2, change_fused)
        """
        B, C, H, W = sem_t1.shape
        
        # Flatten to tokens: [B, C, H, W] -> [B, H*W, C]
        sem_t1_flat = sem_t1.flatten(2).transpose(1, 2)  # [B, HW, C]
        sem_t2_flat = sem_t2.flatten(2).transpose(1, 2)
        chg_flat = change_fused.flatten(2).transpose(1, 2)
        
        # Average semantic tokens from T1 and T2
        sem_avg = (sem_t1_flat + sem_t2_flat) / 2.0  # [B, HW, C]
        
        # Normalize
        sem_norm = self.norm_sem(sem_avg)
        chg_norm = self.norm_chg(chg_flat)
        
        if self.use_mamba:
            # Mamba-style mixing (reshape needed for ConvMamba)
            sem_norm_2d = sem_norm.transpose(1, 2).reshape(B, C, H, W)
            chg_norm_2d = chg_norm.transpose(1, 2).reshape(B, C, H, W)
            
            # Mix: semantic informed by change
            sem_enhanced_2d = sem_norm_2d + self.chg_to_sem(chg_norm_2d)
            # Mix: change informed by semantic
            chg_enhanced_2d = chg_norm_2d + self.sem_to_chg(sem_norm_2d)
            
            sem_enhanced = sem_enhanced_2d.flatten(2).transpose(1, 2)
            chg_enhanced = chg_enhanced_2d.flatten(2).transpose(1, 2)
        else:
            # Cross-attention: semantic queries change context
            sem_enhanced, _ = self.sem_to_chg_attn(
                query=sem_norm, key=chg_norm, value=chg_norm
            )
            sem_enhanced = sem_avg + sem_enhanced  # Residual
            
            # Cross-attention: change queries semantic context
            chg_enhanced, _ = self.chg_to_sem_attn(
                query=chg_norm, key=sem_norm, value=sem_norm
            )
            chg_enhanced = chg_flat + chg_enhanced  # Residual
        
        # FFN refinement
        sem_enhanced = sem_enhanced + self.ffn_sem(self.norm_ffn_sem(sem_enhanced))
        chg_enhanced = chg_enhanced + self.ffn_chg(self.norm_ffn_chg(chg_enhanced))
        
        # Reshape back to feature maps: [B, HW, C] -> [B, C, H, W]
        sem_enhanced = sem_enhanced.transpose(1, 2).reshape(B, C, H, W)
        chg_enhanced = chg_enhanced.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply enhancement to both T1 and T2 semantic features
        sem_t1_out = sem_t1 + sem_enhanced
        sem_t2_out = sem_t2 + sem_enhanced
        change_out = change_fused + chg_enhanced
        
        return sem_t1_out, sem_t2_out, change_out

class ModifiedSRCMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, groups=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.groups = groups
        self.norm = nn.LayerNorm(input_dim)

        # Grouped ConvMamba (split channels across groups)
        self.mambas = nn.ModuleList([
            ConvMamba(d_model=input_dim // groups, d_state=d_state, d_conv=d_conv, expand=expand, bimamba_type="v2")
            for _ in range(groups)
        ])

        self.gate_proj = nn.Linear(input_dim, input_dim)
        self.pos_enc = ConvPosEnc(input_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4096, input_dim))  # Max 64x64 tokens (safe default)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.pos_enc(x)
        x = x.reshape(B, C, -1).transpose(1, 2)  
        pos_embed = F.interpolate(
            self.pos_embed.transpose(1, 2).reshape(1, self.input_dim, int(self.pos_embed.shape[1] ** 0.5), -1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).reshape(1, self.input_dim, -1).transpose(1, 2)  # Shape: [1, H*W, C]
        x = x + pos_embed[:, :x.shape[1], :]

        x_norm = self.norm(x)

        # Grouped Mamba
        chunks = x_norm.chunk(self.groups, dim=-1)
        out_chunks = [m(chunk) for m, chunk in zip(self.mambas, chunks)]
        x_mamba = torch.cat(out_chunks, dim=-1)

        # Gated residual
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x_out = gate * x_mamba + (1 - gate) * x

        x_out = self.proj(x_out)
        return x_out.transpose(1, 2).reshape(B, self.output_dim, H, W)

def get_srcm_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1, conv_mode: str = "deepwise"
):
    srcm_layer = ModifiedSRCMLayer(input_dim=in_channels, output_dim=out_channels)  # Removed conv_mode
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(srcm_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return srcm_layer


class SRCMBlock(nn.Module):

    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            norm: tuple | str,
            kernel_size: int = 3,
            conv_mode: str = "deepwise",
            act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        # print(conv_mode)
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )
        self.conv2 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )
        self.res_scale = nn.Parameter(torch.tensor(1.0))  # residual scaling
        self.drop = nn.Dropout2d(p=0.1) 
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//8, in_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        x = self.drop(x)
        x = x * self.se(x)
        return identity + self.res_scale * x

class CDMamba_seg_cd(nn.Module):
    """
    Segmentation and Change Detection Mamba-based Model.
    - Outputs segmentation for T1, T2, and change map.
    - num_classes: number of semantic classes.
    - If use_change_head=True, outputs per-pixel change logits.
    """

    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 16,
            in_channels: int = 1,
            num_classes: int = 7,
            use_change_head: bool = True,  # <-- NEW: output change map
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
        self.num_classes = num_classes
        self.use_change_head = use_change_head
        self.use_change_gating = use_change_gating
        self.change_gate_alpha = change_gate_alpha
        self.change_gate_beta = change_gate_beta
        self.change_gate_mode = change_gate_mode  # "additive" or "multiplicative"
        self.use_interaction_block = use_interaction_block

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        self.up_conv_mode = up_conv_mode
        self.conv_mode = conv_mode
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.srcm_encoder_layers = self._make_srcm_encoder_layers()
        self.srcm_decoder_layers, self.up_samples = self._make_srcm_decoder_layers()
        self.srcm_decoder_layers_seg_t1, self.up_samples_seg_t1 = self._make_srcm_decoder_layers()
        self.srcm_decoder_layers_seg_t2, self.up_samples_seg_t2 = self._make_srcm_decoder_layers()
        # Dedicated change decoder operating on fused multi-scale features
        self.srcm_decoder_layers_change, self.up_samples_change = self._make_srcm_decoder_layers()

        # ---- Bottleneck context (ASPP-lite / dilated conv stack) ----
        bottleneck_channels = init_filters * (2 ** (len(blocks_down) - 1))  # e.g., 16 * 2**3 = 128 with your defaults
 
        if self.spatial_dims == 2:
            norm_fn = lambda c: get_norm_layer(name=self.norm, spatial_dims=2, channels=c)
            self.context = ContextBlock2D(bottleneck_channels, norm_fn, self.act_mod)
        else:
            self.context = nn.Identity()  # or implement ContextBlock3D

        # Cross-temporal fusion modules: one per encoder scale
        # Encoder produces channels: init_filters * 2**i for i in [0..len(blocks_down)-1]
        self.fuse_scales = nn.ModuleList([
            CrossTemporalFusion(init_filters * (2 ** i)) for i in range(len(blocks_down))
        ])
        # Fusion at bottleneck level
        self.fuse_bottleneck = CrossTemporalFusion(bottleneck_channels)
        
        # Semantic-Change Interaction Block at bottleneck
        if self.use_interaction_block:
            self.interaction_block = SemanticChangeInteractionBlock(
                channels=bottleneck_channels,
                num_heads=interaction_num_heads,
                use_mamba=interaction_use_mamba
            )

        # --- SEGMENTATION HEADS ---
        # Each head outputs num_classes channels
        self.seg_head_t1 = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )
        self.seg_head_t2 = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )
        
        if self.use_change_head:
            # Output 2 channels: [no-change, change]
            # Now expects features from the dedicated change decoder (C = init_filters)
            self.change_head = nn.Sequential(
                get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
                self.act_mod,
                get_conv_layer(self.spatial_dims, self.init_filters, 2, kernel_size=1, bias=True),
            )

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_srcm_encoder_layers(self):
        srcm_encoder_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, conv_mode = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm, self.conv_mode)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            downsample_mamba = (
                get_srcm_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2, conv_mode=conv_mode)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[SRCMBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, conv_mode=conv_mode) for _ in range(item)]
            )
            srcm_encoder_layers.append(down_layer)
        return srcm_encoder_layers

    def _make_srcm_decoder_layers(self):
        srcm_decoder_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        Block_up = SRCMBlock
        n_up = len(blocks_up)
        # in _make_srcm_decoder_layers():
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)          # e.g., 128, 64, 32 at the start
            cat_channels = (sample_in_channels // 2) * 2            # concat of up and skip
            srcm_decoder_layers.append(
                nn.Sequential(
                    get_conv_layer(spatial_dims, cat_channels, sample_in_channels // 2, kernel_size=3, stride=1),
                    Block_up(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act, conv_mode=self.up_conv_mode)
                )
            )
            up_samples.append(
                nn.Sequential(
                    get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                    get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                )
            )

        return srcm_decoder_layers, up_samples

    # removed _make_final_conv (not used in segmentation-only)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.srcm_encoder_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def _decode_with_layers(
        self,
        x: torch.Tensor,
        down_x: list[torch.Tensor],
        up_samples: nn.ModuleList,
        decoder_layers: nn.ModuleList,
    ) -> torch.Tensor:
        """
        down_x is expected to be reversed before calling this:
        down_x[0] = bottleneck feature (same scale as x)
        down_x[1:] = skip features from deep -> shallow
        """
        skips = down_x[1:]  # exclude bottleneck

        interp_mode = "bilinear" if self.spatial_dims == 2 else "trilinear"

        for i, (up, upl) in enumerate(zip(up_samples, decoder_layers)):
            x_up = up(x)
            target = skips[i]

            if x_up.shape[2:] != target.shape[2:]:
                x_up = F.interpolate(
                    x_up, size=target.shape[2:], mode=interp_mode, align_corners=False
                )

            x = torch.cat([x_up, target], dim=1)
            x = upl(x)

        return x



    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        return self._decode_with_layers(x, down_x, self.up_samples, self.srcm_decoder_layers)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None):
        """
        Returns:
            If x2 is None (single image mode):
                seg_logits_t1: [B, num_classes, H, W] -- segmentation logits for T1
            
            If x2 is provided (change detection mode):
                seg_logits_t1: [B, num_classes, H, W] -- segmentation logits for T1
                seg_logits_t2: [B, num_classes, H, W] -- segmentation logits for T2
                change_logits: [B, 2, H, W] -- change logits (if use_change_head=True)
        """
        # Single image segmentation mode
        if x2 is None:
            latent, down_x = self.encode(x1)
            # ---- apply bottleneck context here ----
            latent = self.context(latent)  

            down_x.reverse()
            dec = self._decode_with_layers(latent, down_x, self.up_samples, self.srcm_decoder_layers)
            seg_logits_t1 = self.seg_head_t1(dec)
            return seg_logits_t1
            
        # Change detection mode
        else:
            # Encode both images
            x1_latent, down_x1 = self.encode(x1)
            x2_latent, down_x2 = self.encode(x2)
            
            # Apply bottleneck context
            x1_latent = self.context(x1_latent)
            x2_latent = self.context(x2_latent)
            
            # Cross-temporal fusion at each encoder scale (multi-scale differencing)
            down_x_fused = []
            for i in range(len(down_x1)):
                x1_i, x2_i = down_x1[i], down_x2[i]
                fused_i = self.fuse_scales[i](x1_i, x2_i)
                down_x_fused.append(fused_i)
            
            # Fuse bottleneck features for change path
            fused_latent = self.fuse_bottleneck(x1_latent, x2_latent)
            
            # Semantic-Change Interaction at bottleneck
            if self.use_interaction_block:
                x1_latent, x2_latent, fused_latent = self.interaction_block(
                    x1_latent, x2_latent, fused_latent
                )
            
            # Decode each path
            down_x_fused.reverse()
            down_x1.reverse()
            down_x2.reverse()
            
            # Decode for T1 and T2 (for segmentation)
            seg1 = self._decode_with_layers(x1_latent, down_x1, self.up_samples_seg_t1, self.srcm_decoder_layers_seg_t1)
            seg2 = self._decode_with_layers(x2_latent, down_x2, self.up_samples_seg_t2, self.srcm_decoder_layers_seg_t2)
            
            if self.use_change_head:
                # Decode dedicated change path
                chg_dec = self._decode_with_layers(fused_latent, down_x_fused, self.up_samples_change, self.srcm_decoder_layers_change)
                change_logits = self.change_head(chg_dec)
                
                # Change-guided gating: force semantic heads to focus on change-relevant regions
                if self.use_change_gating:
                    # Compute soft change probability mask
                    if change_logits.size(1) == 2:
                        # 2-channel output: [no-change, change]
                        change_prob = F.softmax(change_logits, dim=1)[:, 1:2]  # [B, 1, H, W]
                    else:
                        # 1-channel output: sigmoid
                        change_prob = torch.sigmoid(change_logits)  # [B, 1, H, W]
                    
                    # Resize change_prob to match decoder feature spatial size
                    if change_prob.shape[2:] != seg1.shape[2:]:
                        change_prob = F.interpolate(
                            change_prob, 
                            size=seg1.shape[2:], 
                            mode='bilinear' if self.spatial_dims == 2 else 'trilinear',
                            align_corners=False
                        )
                    
                    # Apply gating to decoder features before segmentation heads
                    if self.change_gate_mode == "additive":
                        # Additive gating: dec_gated = dec * (1 + alpha * change_prob)
                        # Amplifies features in changed regions
                        seg1_gated = seg1 * (1.0 + self.change_gate_alpha * change_prob)
                        seg2_gated = seg2 * (1.0 + self.change_gate_alpha * change_prob)
                    else:  # multiplicative
                        # Multiplicative gating: dec_gated = dec * (beta + (1-beta) * change_prob)
                        # Suppresses unchanged regions more strongly (beta ~ 0.2)
                        gate = self.change_gate_beta + (1.0 - self.change_gate_beta) * change_prob
                        seg1_gated = seg1 * gate
                        seg2_gated = seg2 * gate
                    
                    seg_logits_t1 = self.seg_head_t1(seg1_gated)
                    seg_logits_t2 = self.seg_head_t2(seg2_gated)
                else:
                    seg_logits_t1 = self.seg_head_t1(seg1)
                    seg_logits_t2 = self.seg_head_t2(seg2)
                
                return seg_logits_t1, seg_logits_t2, change_logits
            else:
                seg_logits_t1 = self.seg_head_t1(seg1)
                seg_logits_t2 = self.seg_head_t2(seg2)
                return seg_logits_t1, seg_logits_t2

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

    # Test single image segmentation mode
    x = torch.randn(1, 3, 256, 256).to(device)
    seg = model(x)
    print("Single image mode output shape:", seg.shape)
    
    # Test change detection mode
    x1 = torch.randn(1, 3, 256, 256).to(device)
    x2 = torch.randn(1, 3, 256, 256).to(device)
    seg_t1, seg_t2, change = model(x1, x2)
    print("Change detection mode output shapes:")
    print(f"  Segmentation T1: {seg_t1.shape}")
    print(f"  Segmentation T2: {seg_t2.shape}")
    print(f"  Change map: {change.shape}")
