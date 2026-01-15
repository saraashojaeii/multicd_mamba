"""
CDMamba Change Detection Model (Change-Only)
Pure change detection without segmentation heads.
Uses cross-temporal fusion with multi-scale differencing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from monai.networks.layers import Dropout
from models.mamba_customer import ConvMamba


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

        if input_dim != output_dim:
            self.proj = nn.Linear(input_dim, output_dim)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        x_norm = self.norm(x_flat)

        # Split into groups
        chunk_size = C // self.groups
        chunks = torch.split(x_norm, chunk_size, dim=-1)
        out_chunks = [mamba(chunk) for mamba, chunk in zip(self.mambas, chunks)]
        out = torch.cat(out_chunks, dim=-1)

        out = self.proj(out)
        out = out.transpose(1, 2).view(B, self.output_dim, H, W)
        return out


def get_srcm_layer(spatial_dims, in_channels, out_channels, stride=1, conv_mode="deepwise"):
    if spatial_dims == 2:
        if stride == 2:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return ModifiedSRCMLayer(in_channels, out_channels)
    else:
        raise NotImplementedError("Only 2D is supported")


class SRCMBlock(nn.Module):
    def __init__(self, spatial_dims, channels, norm, act, conv_mode="deepwise"):
        super().__init__()
        self.srcm = get_srcm_layer(spatial_dims, channels, channels, stride=1, conv_mode=conv_mode)
        self.norm = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=channels)
        self.act = get_act_layer(act)
        
        # SE module
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        out = self.srcm(x)
        out = self.norm(out)
        out = self.act(out)
        
        # SE attention
        se_weight = self.se(out)
        out = out * se_weight
        
        # Residual
        out = out + identity
        return out


class CDMamba_change(nn.Module):
    """
    Pure Change Detection Mamba-based Model.
    - Only outputs binary change map (no segmentation).
    - Uses cross-temporal fusion at multiple scales.
    - num_classes: number of change classes (typically 2: no-change, change).
    """

    def __init__(
            self,
            spatial_dims: int = 2,
            init_filters: int = 16,
            in_channels: int = 3,
            num_classes: int = 2,  # Binary change: [no-change, change]
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
    ):
        super().__init__()
        self.num_classes = num_classes

        if spatial_dims != 2:
            raise ValueError("`spatial_dims` must be 2 for this model.")
        
        self.up_conv_mode = up_conv_mode
        self.conv_mode = conv_mode
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        
        # Encoder
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.srcm_encoder_layers = self._make_srcm_encoder_layers()
        
        # Single decoder for change detection
        self.srcm_decoder_layers, self.up_samples = self._make_srcm_decoder_layers()

        # Bottleneck context
        bottleneck_channels = init_filters * (2 ** (len(blocks_down) - 1))
        
        Conv = nn.Conv2d
        Pool = nn.AdaptiveAvgPool2d

        self.context = nn.Sequential(
            Conv(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=bottleneck_channels),
            self.act_mod,

            Conv(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=bottleneck_channels),
            self.act_mod,

            Pool(1),
            Conv(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False),
            self.act_mod,
        )

        # Cross-temporal fusion modules
        self.fuse_scales = nn.ModuleList([
            CrossTemporalFusion(init_filters * (2 ** i)) for i in range(len(blocks_down))
        ])
        self.fuse_bottleneck = CrossTemporalFusion(bottleneck_channels)

        # Change head - outputs num_classes channels
        self.change_head = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_srcm_encoder_layers(self):
        srcm_encoder_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, conv_mode = (
            self.blocks_down, self.spatial_dims, self.init_filters, self.norm, self.conv_mode
        )
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
        
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            cat_channels = (sample_in_channels // 2) * 2
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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.srcm_encoder_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def _decode_with_layers(self, x: torch.Tensor, down_x: list[torch.Tensor],
                             up_samples: nn.ModuleList, decoder_layers: nn.ModuleList) -> torch.Tensor:
        """Generic decoder that operates on provided up-sample and decoder layer lists."""
        for i, (up, upl) in enumerate(zip(up_samples, decoder_layers)):
            x_up = up(x)
            target = down_x[i + 1]
            if x_up.shape[2:] != target.shape[2:]:
                x_up = F.interpolate(x_up, size=target.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x_up, target], dim=1)
            x = upl(x)
        return x

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Args:
            x1: T1 image [B, C, H, W]
            x2: T2 image [B, C, H, W]
        
        Returns:
            change_logits: [B, num_classes, H, W] - change detection logits
        """
        # Encode both images
        x1_latent, down_x1 = self.encode(x1)
        x2_latent, down_x2 = self.encode(x2)
        
        # Apply bottleneck context
        x1_latent = self.context(x1_latent)
        x2_latent = self.context(x2_latent)
        
        # Cross-temporal fusion at each encoder scale
        down_x_fused = []
        for i in range(len(down_x1)):
            x1_i, x2_i = down_x1[i], down_x2[i]
            fused_i = self.fuse_scales[i](x1_i, x2_i)
            down_x_fused.append(fused_i)
        
        # Reverse for decoder
        down_x_fused.reverse()
        
        # Fuse bottleneck and decode
        fused_latent = self.fuse_bottleneck(x1_latent, x2_latent)
        chg_dec = self._decode_with_layers(fused_latent, down_x_fused, self.up_samples, self.srcm_decoder_layers)
        
        # Change head
        change_logits = self.change_head(chg_dec)
        
        return change_logits


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CDMamba_change(
        spatial_dims=2,
        in_channels=3,
        num_classes=2,  # Binary change
        init_filters=16,
        up_conv_mode="deepwise",
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    ).to(device)

    print(f"Model created on {device}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x1 = torch.randn(2, 3, 256, 256).to(device)
    x2 = torch.randn(2, 3, 256, 256).to(device)
    change = model(x1, x2)
    print(f"Change output shape: {change.shape}")  # Expected: [2, 2, 256, 256]
