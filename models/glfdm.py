"""
Global-Local Feature Decoding Module (GLFDM)

Combines dilated convolutions with lightweight vision transformers
in a local-global-local processing sequence for comprehensive
feature reconstruction in the decoder.

F_GLFDM = Conv(DC_{r=4}(ViT(DC_{r=2}(Conv(F_dec)))))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LightweightViTBlock(nn.Module):
    """
    Lightweight Vision Transformer block with single-layer
    multi-head self-attention for global feature interactions.
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')

        # Self-attention with residual
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out

        # FFN with residual
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        return rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)


class GLFDM(nn.Module):
    """
    Global-Local Feature Decoding Module.

    Local-global-local processing sequence:
    1. Conv + DC_{r=2}: fine-grained local patterns
    2. ViT: global feature interactions
    3. DC_{r=4} + Conv: expanded receptive field refinement

    Includes skip connection from encoder for spatial detail preservation.
    """

    def __init__(self, in_channels, skip_channels=None, out_channels=None,
                 num_heads=4, vit_mlp_ratio=2.0):
        super().__init__()
        if skip_channels is None:
            skip_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        # Skip connection fusion
        self.skip_fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        ) if skip_channels > 0 else nn.Identity()

        # Stage 1: Local - Conv + Dilated Conv (r=2)
        self.local_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.dc_r2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # Stage 2: Global - Lightweight ViT
        self.vit = LightweightViTBlock(in_channels, num_heads, vit_mlp_ratio)

        # Stage 3: Local - Dilated Conv (r=4) + Conv
        self.dc_r4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )
        self.local_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Upsample (2x)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip=None):
        """
        Args:
            x: Decoder features [B, C, H, W]
            skip: Encoder skip features [B, C_skip, 2H, 2W] (optional)
        Returns:
            Decoded features [B, C_out, 2H, 2W]
        """
        # Upsample first
        x = self.upsample(x)

        # Fuse with skip connection
        if skip is not None:
            # Ensure spatial alignment
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = self.skip_fuse(torch.cat([x, skip], dim=1))

        # Local-Global-Local processing
        x = self.local_in(x)      # Conv
        x = self.dc_r2(x)         # DC_{r=2}
        x = self.vit(x)           # ViT (global)
        x = self.dc_r4(x)         # DC_{r=4}
        x = self.local_out(x)     # Conv

        return x
