"""
State-Space Global Context Module (SSGCM)

Selective state-space modeling (Mamba) with four-directional SS2D scanning,
Local Scan (LS) Block, SS2D module, and Recurrent Gating (RG) Block.
LGDFM is embedded within SSGCM at multiple positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .lgdfm import LGDFM


class SelectiveSSM(nn.Module):
    """
    Selective State-Space Model (Mamba-style).
    Implements discretized SSM recurrence:
        h[k] = A_bar * h[k-1] + B_bar * x[k]
        y[k] = C * h[k] + D * x[k]
    with input-dependent B, C, Delta (selective mechanism).
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Conv for local context before SSM
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner
        )

        # SSM parameters (input-dependent / selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # A parameter (log-space for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        self.act = nn.SiLU()

    def forward(self, x):
        """
        Args:
            x: [B, L, D] input sequence
        Returns:
            y: [B, L, D] output sequence
        """
        B, L, D = x.shape

        # Input projection -> split into x and z (gate)
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Local conv
        x_ssm = x_ssm.transpose(1, 2)  # [B, d_inner, L]
        x_ssm = self.conv1d(x_ssm)[:, :, :L]  # causal padding
        x_ssm = x_ssm.transpose(1, 2)  # [B, L, d_inner]
        x_ssm = self.act(x_ssm)

        # Selective SSM parameters (input-dependent)
        x_proj = self.x_proj(x_ssm)  # [B, L, 2*N+1]
        B_sel = x_proj[:, :, :self.d_state]  # [B, L, N]
        C_sel = x_proj[:, :, self.d_state:2*self.d_state]  # [B, L, N]
        delta = F.softplus(x_proj[:, :, -1:])  # [B, L, 1]

        # Discretize A
        A = -torch.exp(self.A_log)  # [d_inner, N]

        # Simplified selective scan (sequential for correctness)
        y = self._selective_scan(x_ssm, A, B_sel, C_sel, delta)

        # Skip connection with D
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)

        # Gate with z
        y = y * self.act(z)

        # Output projection
        y = self.out_proj(y)

        return y

    def _selective_scan(self, x, A, B, C, delta):
        """
        Simplified selective scan implementation.
        For production use, replace with CUDA-optimized Mamba kernel.
        """
        B_batch, L, d_inner = x.shape
        N = self.d_state

        # Discretize: A_bar = exp(delta * A)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, d_inner, N]
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, d_inner, N] (broadcast)

        # Adjust dimensions for proper broadcasting
        delta_B = delta_B.expand(-1, -1, d_inner, -1)

        # Sequential scan
        h = torch.zeros(B_batch, d_inner, N, device=x.device, dtype=x.dtype)
        ys = []

        for i in range(L):
            h = delta_A[:, i] * h + delta_B[:, i] * x[:, i].unsqueeze(-1)
            y_i = (h * C[:, i].unsqueeze(1)).sum(dim=-1)  # [B, d_inner]
            ys.append(y_i)

        y = torch.stack(ys, dim=1)  # [B, L, d_inner]
        return y


class SS2D(nn.Module):
    """
    State-Space 2D (SS2D) scanning module.
    Traverses the feature map along four directions
    (L->R, R->L, T->B, B->T) and merges results.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # Four-directional SSMs
        self.ssm_lr = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_rl = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_tb = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_bt = SelectiveSSM(d_model, d_state, d_conv, expand)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 2D feature map
        Returns:
            y: [B, C, H, W] globally contextualized features
        """
        B, C, H, W = x.shape

        # Direction 1: Left-to-Right (row-major)
        x_lr = rearrange(x, 'b c h w -> b (h w) c')

        # Direction 2: Right-to-Left
        x_rl = x_lr.flip(dims=[1])

        # Direction 3: Top-to-Bottom (column-major)
        x_tb = rearrange(x, 'b c h w -> b (w h) c')

        # Direction 4: Bottom-to-Top
        x_bt = x_tb.flip(dims=[1])

        # Apply SSM in each direction
        y_lr = self.ssm_lr(self.norm(x_lr))
        y_rl = self.ssm_rl(self.norm(x_rl)).flip(dims=[1])
        y_tb = self.ssm_tb(self.norm(x_tb))
        y_bt = self.ssm_bt(self.norm(x_bt)).flip(dims=[1])

        # Reshape back to 2D
        y_lr = rearrange(y_lr, 'b (h w) c -> b c h w', h=H, w=W)
        y_rl = rearrange(y_rl, 'b (h w) c -> b c h w', h=H, w=W)
        y_tb = rearrange(y_tb, 'b (w h) c -> b c h w', h=H, w=W)
        y_bt = rearrange(y_bt, 'b (w h) c -> b c h w', h=H, w=W)

        # Element-wise sum of four directions
        y = y_lr + y_rl + y_tb + y_bt

        return y


class LocalScanBlock(nn.Module):
    """LS Block: Local feature pre-processing before SSM scanning."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class RecurrentGatingBlock(nn.Module):
    """
    RG Block: Gated linear unit for controlling information flow.
    Z_RG = σ(W_g·Z + b_g) ⊙ (W_v·Z + b_v)
    """

    def __init__(self, channels):
        super().__init__()
        self.gate_proj = nn.Linear(channels, channels)
        self.value_proj = nn.Linear(channels, channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Gated features [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')

        gate = torch.sigmoid(self.gate_proj(x_flat))
        value = self.value_proj(x_flat)
        out = gate * value

        return rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)


class SSGCM(nn.Module):
    """
    State-Space Global Context Module.

    Pipeline: LS Block -> SS2D -> RG Block
    LGDFM is embedded at multiple positions for local-global fusion.
    """

    def __init__(self, channels, d_state=16, d_conv=4, expand=2, use_lgdfm=True):
        super().__init__()

        # Local Scan Block
        self.ls_block = LocalScanBlock(channels)

        # LGDFM before SS2D (position a: core block)
        self.lgdfm_pre = LGDFM(channels, channels, channels) if use_lgdfm else nn.Identity()

        # SS2D module
        self.ss2d = SS2D(channels, d_state, d_conv, expand)

        # LGDFM after SS2D (position c: residual block)
        self.lgdfm_post = LGDFM(channels, channels, channels) if use_lgdfm else nn.Identity()

        # Recurrent Gating Block
        self.rg_block = RecurrentGatingBlock(channels)

        # Residual projection
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            Globally contextualized features [B, C, H, W]
        """
        identity = x

        # LS Block: local pre-processing
        x = self.ls_block(x)

        # LGDFM (pre-SS2D): local-global fusion enrichment
        x = self.lgdfm_pre(x)

        # SS2D: four-directional selective state-space scanning
        x = self.ss2d(x)

        # LGDFM (post-SS2D): further enrichment
        x = self.lgdfm_post(x)

        # RG Block: gated information flow
        x = self.rg_block(x)

        # Residual connection
        x = self.norm(x + identity)

        return x
