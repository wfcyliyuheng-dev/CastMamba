"""
CastMamba Training Script

Usage:
    python train.py --config configs/default.yaml --dataset_path /path/to/data
    python train.py --task segmentation --dataset mvtec --img_size 256
"""

import os
import argparse
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pathlib import Path

from models import CastMamba
from utils.losses import CastMambaLoss


class DefectDataset(Dataset):
    """
    Generic defect detection dataset.
    Supports MCDD, BDD (YOLO format) and MVTec-AD.

    Expected directory structure for detection:
        dataset_path/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/

    For MVTec-AD:
        dataset_path/
            <category>/
                train/good/
                test/<defect_type>/
                ground_truth/<defect_type>/
    """

    def __init__(self, root, split='train', img_size=640, task='detection', transform=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.task = task
        self.transform = transform

        if task == 'detection':
            img_dir = self.root / 'images' / split
            self.images = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
            self.label_dir = self.root / 'labels' / split
        else:
            # MVTec-AD structure
            if split == 'train':
                img_dir = self.root / 'train' / 'good'
            else:
                img_dir = self.root / 'test'
            self.images = []
            for ext in ('*.png', '*.jpg', '*.bmp'):
                self.images.extend(sorted(img_dir.rglob(ext)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import cv2
        import numpy as np

        img_path = str(self.images[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        if self.task == 'detection':
            label_path = self.label_dir / (self.images[idx].stem + '.txt')
            labels = []
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x, y, w, h = [float(p) for p in parts[1:5]]
                            labels.append([cls_id, x, y, w, h])
            labels = torch.tensor(labels) if labels else torch.zeros((0, 5))
            return img, labels
        else:
            # Segmentation: return image as both input and target for reconstruction
            gt_path = str(self.images[idx]).replace('test', 'ground_truth')
            if os.path.exists(gt_path):
                mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.img_size, self.img_size))
                mask = (mask > 127).astype(np.float32)
                mask = torch.from_numpy(mask).unsqueeze(0)
            else:
                mask = torch.zeros(1, self.img_size, self.img_size)
            return img, mask


def collate_fn(batch):
    """Custom collate for variable-length detection labels."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels)


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        if isinstance(outputs, tuple):
            # Detection head returns (cls, reg)
            loss = criterion(outputs, targets)
        else:
            # Segmentation
            targets = torch.stack(targets).to(device) if isinstance(targets, list) else targets.to(device)
            loss = criterion(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f'  Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}')

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    for images, targets in dataloader:
        images = images.to(device)
        outputs = model(images)

        if isinstance(outputs, tuple):
            loss = criterion(outputs, targets)
        else:
            targets = torch.stack(targets).to(device) if isinstance(targets, list) else targets.to(device)
            loss = criterion(outputs, targets)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description='CastMamba Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'segmentation'])
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--output_dir', type=str, default='runs/train')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Override with CLI args
    epochs = args.epochs or cfg['train']['epochs']
    batch_size = args.batch_size or cfg['train']['batch_size']
    img_size = args.img_size or cfg['train']['img_size']
    lr = args.lr or cfg['train']['lr']

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model
    model = CastMamba(
        in_channels=3,
        base_dim=cfg['model']['base_dim'],
        num_classes=args.num_classes,
        d_state=cfg['model']['d_state'],
        expand=cfg['model']['expand'],
        task=args.task,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'CastMamba parameters: {num_params:.2f}M')

    # Loss
    if args.task == 'detection':
        criterion = CastMambaLoss(
            task='detection',
            lambda_cls=cfg['loss']['lambda_cls'],
            lambda_reg=cfg['loss']['lambda_reg'],
            lambda_dfl=cfg['loss']['lambda_dfl'],
        )
    else:
        criterion = CastMambaLoss(
            task='segmentation',
            lambda_seg1=cfg['seg_loss']['lambda1'],
            lambda_seg2=cfg['seg_loss']['lambda2'],
        )

    # Dataset
    train_dataset = DefectDataset(args.dataset_path, 'train', img_size, args.task)
    val_dataset = DefectDataset(args.dataset_path, 'val', img_size, args.task)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=cfg['hardware']['pin_memory'],
        collate_fn=collate_fn if args.task == 'detection' else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=cfg['hardware']['num_workers'],
        pin_memory=cfg['hardware']['pin_memory'],
        collate_fn=collate_fn if args.task == 'detection' else None,
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=cfg['train']['weight_decay'])

    # Scheduler: linear warmup + cosine annealing
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=cfg['train']['warmup_epochs'])
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - cfg['train']['warmup_epochs'])
    scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                             milestones=[cfg['train']['warmup_epochs']])

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}')

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f'\nStarting training for {epochs} epochs...')
    print(f'  Dataset: {args.dataset_path}')
    print(f'  Task: {args.task}')
    print(f'  Image size: {img_size}')
    print(f'  Batch size: {batch_size}')
    print(f'  Learning rate: {lr}\n')

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch}/{epochs-1} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'LR: {current_lr:.6f} | '
              f'Time: {elapsed:.1f}s')

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
        }
        torch.save(ckpt, os.path.join(args.output_dir, 'last.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, os.path.join(args.output_dir, 'best.pt'))
            print(f'  -> New best model saved (val_loss: {val_loss:.4f})')

    print(f'\nTraining complete. Best val loss: {best_val_loss:.4f}')
    print(f'Checkpoints saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
