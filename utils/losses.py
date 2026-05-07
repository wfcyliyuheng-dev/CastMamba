"""
Loss functions for CastMamba.

Detection: L = λ_cls * L_cls + λ_reg * L_reg + λ_dfl * L_dfl
    - L_cls: Varifocal Loss (VFL)
    - L_reg: Complete IoU (CIoU) Loss
    - L_dfl: Distribution Focal Loss (DFL)

Segmentation: L_seg = λ_1 * L_dice + λ_2 * L_ssim
    - L_dice: Dice Loss
    - L_ssim: SSIM Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss for IoU-aware classification.
    Addresses foreground-background class imbalance with
    IoU-aware soft labels.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target_score, target_cls):
        """
        Args:
            pred: Predicted classification scores [B, N, C]
            target_score: Target IoU scores [B, N] (0 for bg, IoU for fg)
            target_cls: Target class labels [B, N] (class index)
        """
        pred_sigmoid = pred.sigmoid()

        # Positive samples: q * BCE
        pos_mask = target_score > 0
        pos_loss = target_score[pos_mask] * F.binary_cross_entropy_with_logits(
            pred[pos_mask], target_score[pos_mask].unsqueeze(-1).expand_as(pred[pos_mask]),
            reduction='none'
        ).sum(dim=-1)

        # Negative samples: alpha * p^gamma * BCE
        neg_mask = ~pos_mask
        neg_loss = self.alpha * pred_sigmoid[neg_mask].pow(self.gamma) * \
                   F.binary_cross_entropy_with_logits(
                       pred[neg_mask],
                       torch.zeros_like(pred[neg_mask]),
                       reduction='none'
                   ).sum(dim=-1)

        total = pos_loss.sum() + neg_loss.sum()
        num_pos = pos_mask.sum().clamp(min=1)

        return total / num_pos


class CIoULoss(nn.Module):
    """
    Complete IoU Loss for bounding box regression.
    L_reg = 1 - IoU + ρ²(b, b^gt)/c² + βv
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] (x1, y1, x2, y2)
            target_boxes: [N, 4] (x1, y1, x2, y2)
        """
        # Intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        # Union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area

        iou = inter_area / union_area.clamp(min=1e-6)

        # Enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

        # Distance term ρ²/c²
        pred_center = torch.stack([
            (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2,
            (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        ], dim=-1)
        target_center = torch.stack([
            (target_boxes[:, 0] + target_boxes[:, 2]) / 2,
            (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        ], dim=-1)

        rho2 = ((pred_center - target_center) ** 2).sum(dim=-1)
        c2 = ((enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2).clamp(min=1e-6)

        # Aspect ratio term v and β
        pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1e-6)
        pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1e-6)
        target_w = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=1e-6)
        target_h = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=1e-6)

        v = (4 / (math.pi ** 2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
        with torch.no_grad():
            beta = v / ((1 - iou) + v + 1e-6)

        ciou_loss = 1 - iou + rho2 / c2 + beta * v

        return ciou_loss.mean()


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss for fine-grained bbox regression.
    Learns discrete distribution of box boundaries.
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred, target):
        """
        Args:
            pred: [N, 4*(reg_max+1)] distribution logits
            target: [N, 4] continuous regression targets
        """
        B = pred.shape[0]
        pred = pred.reshape(-1, self.reg_max + 1)
        target = target.reshape(-1)

        # Discretize target
        target_left = target.long().clamp(0, self.reg_max - 1)
        target_right = (target_left + 1).clamp(max=self.reg_max)
        weight_right = target - target_left.float()
        weight_left = 1 - weight_right

        loss = F.cross_entropy(pred, target_left, reduction='none') * weight_left + \
               F.cross_entropy(pred, target_right, reduction='none') * weight_right

        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice


class SSIMLoss(nn.Module):
    """Structural Similarity Index Loss for segmentation quality."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _gaussian_window(self, channels, device):
        coords = torch.arange(self.window_size, dtype=torch.float32, device=device)
        coords -= self.window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * self.sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) * g.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0).expand(channels, 1, -1, -1)
        return window

    def forward(self, pred, target):
        C = pred.shape[1]
        window = self._gaussian_window(C, pred.device)
        pad = self.window_size // 2

        mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
        mu_target = F.conv2d(target, window, padding=pad, groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_cross

        ssim_map = ((2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) * (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1 - ssim_map.mean()


class SegmentationLoss(nn.Module):
    """Combined segmentation loss: L_seg = λ1 * L_dice + λ2 * L_ssim"""

    def __init__(self, lambda1=1.0, lambda2=0.5):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.ssim_loss = SSIMLoss()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, pred, target):
        return self.lambda1 * self.dice_loss(pred, target) + \
               self.lambda2 * self.ssim_loss(pred, target)


class CastMambaLoss(nn.Module):
    """
    Composite loss for CastMamba.
    Detection: L = λ_cls * VFL + λ_reg * CIoU + λ_dfl * DFL
    Segmentation: L = λ1 * Dice + λ2 * SSIM
    """

    def __init__(self, task='detection',
                 lambda_cls=0.5, lambda_reg=7.5, lambda_dfl=1.5,
                 lambda_seg1=1.0, lambda_seg2=0.5):
        super().__init__()
        self.task = task

        if task == 'detection':
            self.cls_loss = VarifocalLoss()
            self.reg_loss = CIoULoss()
            self.dfl_loss = DistributionFocalLoss()
            self.lambda_cls = lambda_cls
            self.lambda_reg = lambda_reg
            self.lambda_dfl = lambda_dfl
        else:
            self.seg_loss = SegmentationLoss(lambda_seg1, lambda_seg2)

    def forward(self, predictions, targets):
        if self.task == 'detection':
            cls_pred, reg_pred = predictions
            cls_target, reg_target = targets
            loss = self.lambda_cls * self.cls_loss(cls_pred, cls_target['scores'], cls_target['labels']) + \
                   self.lambda_reg * self.reg_loss(reg_pred, reg_target) + \
                   self.lambda_dfl * self.dfl_loss(reg_pred, reg_target)
            return loss
        else:
            return self.seg_loss(predictions, targets)
