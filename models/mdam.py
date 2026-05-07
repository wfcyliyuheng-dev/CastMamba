"""
Micro-Defect Attention Module (MDAM)

Multi-scale dilated convolutions with Bilinear Attention Module (BAM)
for fine-grained defect feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearAttentionModule(nn.Module):
    """
    Bilinear Attention Module (BAM).
    Computes spatial attention via bilinear query-key interaction
    with linear complexity O(H'W') instead of O(H'^2 W'^2).
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid_channels = channels // reduction

        self.pwconv_q = nn.Conv2d(channels, mid_channels, 1)
        self.pwconv_k = nn.Conv2d(channels, mid_channels, 1)
        self.pwconv_v = nn.Conv2d(channels, channels, 1)

        self.fc_q = nn.Linear(mid_channels, mid_channels)
        self.fc_k = nn.Linear(mid_channels, mid_channels)

    def forward(self, f_agg, f_orig):
        """
        Args:
            f_agg: Aggregated multi-scale features [B, C, H, W]
            f_orig: Original input features [B, C, H, W]
        Returns:
            Attention-modulated features [B, C, H, W]
        """
        B, C, H, W = f_agg.shape

        # Query and Key projections
        q = self.pwconv_q(f_agg)  # [B, C//r, H, W]
        k = self.pwconv_k(f_agg)  # [B, C//r, H, W]

        # Global average pool -> FC for low-rank projection
        q = q.mean(dim=[2, 3])  # [B, C//r]
        k = k.mean(dim=[2, 3])  # [B, C//r]

        q = self.fc_q(q)  # [B, C//r]
        k = self.fc_k(k)  # [B, C//r]

        # Bilinear attention: outer product + softmax
        attn = torch.bmm(q.unsqueeze(2), k.unsqueeze(1))  # [B, C//r, C//r]
        attn = F.softmax(attn.view(B, -1), dim=-1).view(B, q.size(1), k.size(1))

        # Aggregate attention to spatial map
        attn_map = attn.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True)  # [B, 1, 1]
        attn_map = attn_map.unsqueeze(-1).expand(B, 1, H, W)  # [B, 1, H, W]
        attn_map = torch.sigmoid(attn_map)

        # Value branch
        v = self.pwconv_v(f_orig)  # [B, C, H, W]

        return attn_map * v


class MDAM(nn.Module):
    """
    Micro-Defect Attention Module.

    Combines multi-scale dilated convolutions with dilation rates r={2,4,4}
    and a Bilinear Attention Module for micro-defect feature extraction.
    Output: F_MDAM = F + A ⊙ V (residual connection)
    """

    def __init__(self, in_channels, out_channels, dilation_rates=(2, 4, 4)):
        super().__init__()
        self.dilation_rates = dilation_rates

        # Multi-scale dilated convolution branches
        self.dilated_convs = nn.ModuleList()
        for r in dilation_rates:
            self.dilated_convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.GELU(),
                    nn.Conv2d(in_channels, in_channels, 1, bias=False),
                    nn.BatchNorm2d(in_channels),
                )
            )

        # Aggregation conv (merge 3 branches)
        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_channels * len(dilation_rates), in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
        )

        # Bilinear Attention Module
        self.bam = BilinearAttentionModule(in_channels)

        # Downsample + channel expansion
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Residual projection if channels change
        self.residual_proj = None
        if in_channels != out_channels:
            self.residual_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """
        Args:
            x: Input feature map [B, C_in, H, W]
        Returns:
            Output feature map [B, C_out, H/2, W/2]
        """
        identity = x

        # Multi-scale dilated convolution branches
        branch_outputs = []
        for dilated_conv in self.dilated_convs:
            branch_outputs.append(dilated_conv(x))

        # Aggregate branches
        f_agg = self.agg_conv(torch.cat(branch_outputs, dim=1))

        # Bilinear Attention
        attn_out = self.bam(f_agg, x)

        # Residual connection: F_MDAM = F + A ⊙ V
        x = identity + attn_out

        # Downsample
        out = self.downsample(x)

        return out
