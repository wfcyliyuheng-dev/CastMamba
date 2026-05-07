"""
Local-Global Dual-Branch Fusion Module (LGDFM)

Dual patch-size local-global attention branches (p=2, p=4) with
channel-spatial attention refinement and RepConv fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalAttention(nn.Module):
    """
    Patch-based local-global attention at a given patch size p.
    Partitions feature map into p×p patches, computes mean-pooled
    attention within each patch, followed by channel and spatial refinement.
    """

    def __init__(self, channels, patch_size=2):
        super().__init__()
        self.patch_size = patch_size

        # Patch-level attention
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Channel attention (GAP -> FC -> Sigmoid)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )

        # Spatial attention (Conv -> Sigmoid)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Refined features [B, C, H, W]
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Ensure divisible by patch size
        assert H % p == 0 and W % p == 0, \
            f"Feature map size ({H}, {W}) must be divisible by patch size {p}"

        # Unfold into patches: [B, C, H/p, p, W/p, p]
        x_patches = x.reshape(B, C, H // p, p, W // p, p)

        # Mean pool within patches: [B, C, H/p, W/p]
        x_mean = x_patches.mean(dim=[3, 5])

        # Patch-level attention
        attn = self.attn_conv(x_mean)

        # Upsample back to original resolution
        attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)

        # Apply patch attention
        x_refined = x * attn

        # Channel attention: F_ch = σ(GAP(F')) ⊙ F'
        ch_attn = self.channel_attn(x_refined)  # [B, C]
        x_ch = x_refined * ch_attn.unsqueeze(-1).unsqueeze(-1)

        # Spatial attention: F_sp = σ(Conv(F_ch)) ⊙ F_ch
        sp_attn = self.spatial_attn(x_ch)  # [B, 1, H, W]
        x_sp = x_ch * sp_attn

        return x_sp


class RepConv(nn.Module):
    """
    Reparameterizable Convolution (RepConv).
    Uses multi-branch (3x3 + 1x1 + identity) during training,
    fused into single 3x3 during inference.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv3x3(x) + self.conv1x1(x) + x)


class LGDFM(nn.Module):
    """
    Local-Global Dual-Branch Fusion Module.

    Two branches with patch sizes p=2 (fine local) and p=4 (broader global),
    concatenated and refined through Conv-RepConv-Conv.
    F_LGDFM = Conv(RepConv(Conv([F_sp^{p=2} || F_sp^{p=4}])))
    """

    def __init__(self, in_channels1, in_channels2=None, out_channels=None,
                 patch_sizes=(2, 4)):
        super().__init__()
        if in_channels2 is None:
            in_channels2 = in_channels1
        if out_channels is None:
            out_channels = in_channels1

        # Alignment convolutions
        self.align1 = nn.Sequential(
            nn.Conv2d(in_channels1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.align2 = nn.Sequential(
            nn.Conv2d(in_channels2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Fusion pointwise conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Dual-branch local-global attention
        self.branches = nn.ModuleList([
            LocalGlobalAttention(out_channels, p) for p in patch_sizes
        ])

        # Merge: Conv -> RepConv -> Conv
        self.merge = nn.Sequential(
            nn.Conv2d(out_channels * len(patch_sizes), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.repconv = RepConv(out_channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, f1, f2=None):
        """
        Args:
            f1: Feature map 1 [B, C1, H, W]
            f2: Feature map 2 [B, C2, H, W] (optional, same as f1 if None)
        Returns:
            Fused features [B, C_out, H, W]
        """
        if f2 is None:
            f2 = f1

        # Align channels
        f1 = self.align1(f1)
        f2 = self.align2(f2)

        # Element-wise addition fusion
        f_fuse = self.fuse_conv(f1 + f2)

        # Dual-branch processing
        branch_outs = [branch(f_fuse) for branch in self.branches]

        # Concatenate and merge
        x = self.merge(torch.cat(branch_outs, dim=1))
        x = self.repconv(x)
        x = self.out_conv(x)

        return x
