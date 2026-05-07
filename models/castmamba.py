"""
CastMamba: Efficient Defect Detection in Mechanical Castings
via State-Space Modeling and Multi-Scale Attention

Full encoder-bottleneck-decoder architecture:
- Encoder: ConvModule + 4 × MDAM blocks
- Bottleneck: SSGCM (with embedded LGDFM)
- Decoder: 2 × SSGCM + 2 × GLFDM blocks
- Head: Detection or Segmentation head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mdam import MDAM
from .ssgcm import SSGCM
from .lgdfm import LGDFM
from .glfdm import GLFDM


class ConvModule(nn.Module):
    """
    ConvModule: Three cascaded convolutional layers with BN and GELU.
    Used at both input and output stages.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class DetectionHead(nn.Module):
    """
    Decoupled detection head with separate classification and regression branches.
    Each branch: two 3×3 conv layers.
    """

    def __init__(self, in_channels, num_classes, num_anchors=1):
        super().__init__()
        # Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, num_classes * num_anchors, 1),
        )

        # Regression branch (x, y, w, h)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, 4 * num_anchors, 1),
        )

    def forward(self, x):
        cls_out = self.cls_branch(x)
        reg_out = self.reg_branch(x)
        return cls_out, reg_out


class SegmentationHead(nn.Module):
    """
    Segmentation head for anomaly detection.
    1×1 convolution + sigmoid for pixel-level anomaly probability.
    """

    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x)


class CastMamba(nn.Module):
    """
    CastMamba: Full encoder-decoder architecture.

    Architecture:
        Input -> ConvModule -> MDAM×4 (encoder) -> SSGCM (bottleneck)
              -> SSGCM×2 + GLFDM×2 (decoder) -> ConvModule -> Head -> Output

    Channel progression (base_dim=C):
        H/2×W/2×C -> H/4×W/4×2C -> H/8×W/8×4C -> H/16×W/16×8C -> H/16×W/16×16C

    Args:
        in_channels: Input image channels (default: 3)
        base_dim: Base channel dimension C (default: 64)
        num_classes: Number of detection classes (default: 6 for MCDD)
        d_state: SSM state dimension N (default: 16)
        task: 'detection' or 'segmentation'
    """

    def __init__(self, in_channels=3, base_dim=64, num_classes=6,
                 d_state=16, expand=2, task='detection'):
        super().__init__()
        self.task = task
        C = base_dim
        dims = [C, 2*C, 4*C, 8*C, 16*C]  # [64, 128, 256, 512, 1024]

        # ============ Encoder ============
        # Input ConvModule: 3 -> C, stride 2
        self.input_conv = ConvModule(in_channels, dims[0], stride=2)

        # 4 cascaded MDAM blocks
        self.encoder = nn.ModuleList([
            MDAM(dims[0], dims[1]),  # H/2->H/4, C->2C
            MDAM(dims[1], dims[2]),  # H/4->H/8, 2C->4C
            MDAM(dims[2], dims[3]),  # H/8->H/16, 4C->8C
            MDAM(dims[3], dims[4]),  # H/16->H/16 (no spatial downsample for last), 8C->16C
        ])

        # Fix last MDAM: no downsampling (stays at H/16)
        self.encoder[3] = nn.Sequential(
            MDAM(dims[3], dims[3]),  # Keep at 8C with downsample
            nn.Conv2d(dims[3], dims[4], 1, bias=False),  # Channel expand to 16C
            nn.BatchNorm2d(dims[4]),
            nn.GELU(),
        )

        # ============ Bottleneck ============
        self.bottleneck = SSGCM(dims[4], d_state=d_state, expand=expand, use_lgdfm=True)

        # ============ Decoder ============
        # Decoder SSGCM blocks
        self.dec_ssgcm1 = SSGCM(dims[4], d_state=d_state, expand=expand, use_lgdfm=True)
        self.dec_ssgcm2 = SSGCM(dims[3], d_state=d_state, expand=expand, use_lgdfm=True)

        # Channel reduction before decoder SSGCM2
        self.dec_reduce1 = nn.Sequential(
            nn.Conv2d(dims[4], dims[3], 1, bias=False),
            nn.BatchNorm2d(dims[3]),
            nn.GELU(),
        )

        # GLFDM decoder blocks (with skip connections)
        self.dec_glfdm1 = GLFDM(dims[3], skip_channels=dims[2], out_channels=dims[2])
        self.dec_glfdm2 = GLFDM(dims[2], skip_channels=dims[1], out_channels=dims[1])

        # Output ConvModule
        self.output_conv = ConvModule(dims[1], dims[0])

        # ============ Head ============
        if task == 'detection':
            self.head = DetectionHead(dims[0], num_classes)
        else:
            self.head = SegmentationHead(dims[0])

    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            Detection: (cls_out, reg_out) at [B, num_cls, H/2, W/2] and [B, 4, H/2, W/2]
            Segmentation: anomaly_map [B, 1, H/2, W/2]
        """
        # ============ Encoder ============
        # Input ConvModule: [B, 3, H, W] -> [B, C, H/2, W/2]
        x0 = self.input_conv(x)

        # MDAM blocks (store for skip connections)
        skips = []
        feat = x0
        for i, mdam in enumerate(self.encoder):
            skips.append(feat)  # Save before downsampling
            feat = mdam(feat)

        # skips: [C@H/2, 2C@H/4, 4C@H/8, 8C@H/16]
        # feat: [16C@H/32]

        # ============ Bottleneck ============
        feat = self.bottleneck(feat)

        # ============ Decoder ============
        # Decoder SSGCM 1
        feat = self.dec_ssgcm1(feat)

        # Reduce channels: 16C -> 8C
        feat = self.dec_reduce1(feat)

        # Upsample + skip from encoder stage 3
        feat = F.interpolate(feat, size=skips[3].shape[2:], mode='bilinear', align_corners=False)
        feat = feat + skips[3]  # Skip connection

        # Decoder SSGCM 2
        feat = self.dec_ssgcm2(feat)

        # GLFDM 1: 8C -> 4C with skip from stage 2
        feat = self.dec_glfdm1(feat, skip=skips[2])

        # GLFDM 2: 4C -> 2C with skip from stage 1
        feat = self.dec_glfdm2(feat, skip=skips[1])

        # Output ConvModule: 2C -> C
        feat = self.output_conv(feat)

        # ============ Head ============
        out = self.head(feat)

        return out


def build_castmamba(num_classes=6, base_dim=64, task='detection', **kwargs):
    """Factory function to build CastMamba model."""
    return CastMamba(
        in_channels=3,
        base_dim=base_dim,
        num_classes=num_classes,
        d_state=kwargs.get('d_state', 16),
        expand=kwargs.get('expand', 2),
        task=task,
    )
