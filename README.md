# CastMamba

**CastMamba: Efficient Defect Detection in Mechanical Castings via State-Space Modeling and Multi-Scale Attention**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

CastMamba is a novel encoder–decoder architecture for mechanical casting defect detection that integrates selective state-space modeling (Mamba) with multi-scale attention mechanisms.

<p align="center">
  <img src="assets/architecture.png" width="90%">
</p>

### Key Modules

| Module | Description |
|--------|-------------|
| **MDAM** | Micro-Defect Attention Module with multi-scale dilated convolutions (r∈{2,4,4}) and Bilinear Attention |
| **SSGCM** | State-Space Global Context Module with four-directional SS2D scanning and recurrent gating |
| **LGDFM** | Local-Global Dual-Branch Fusion Module with dual patch-size (p=2, p=4) attention |
| **GLFDM** | Global-Local Feature Decoding Module combining dilated convolutions with lightweight ViT |

### Performance

| Dataset | Metric | CastMamba | Best Baseline |
|---------|--------|-----------|---------------|
| **MCDD** | mAP@50 | **88.4%** | 86.2% (YOLOv13) |
| **BDD** | mAP@50 | **91.2%** | 88.7% (YOLOv13) |
| **MVTec-AD** | I-AUROC | **99.4%** | 99.1% (PatchCore) |

- **Parameters**: 12.8M
- **FLOPs**: 32.4G
- **Throughput**: 46.3 FPS (RTX 4090)

## Installation

```bash
git clone https://github.com/wfcyliyuheng-dev/CastMamba.git
cd CastMamba
pip install -r requirements.txt
```

## Datasets

- **MCDD**: [Mechanical Casting Defect Detection Dataset](https://universe.roboflow.com/datasearch-aankn/defects-detection-zurkz/dataset/1)
- **BDD**: [Bearing Defect Dataset](https://universe.roboflow.com/zyb-aeme4/yolo11n_bear-gk2v5/dataset/1)
- **MVTec-AD**: [MVTec Anomaly Detection](https://www.mvtec.com/company/research/datasets/mvtec-ad)

### Dataset Structure

```
datasets/
├── MCDD/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── BDD/
│   └── (same structure)
└── MVTec-AD/
    ├── bottle/
    │   ├── train/good/
    │   ├── test/<defect_type>/
    │   └── ground_truth/<defect_type>/
    └── ...
```

## Training

### MCDD (6 classes)
```bash
python train.py \
  --dataset_path datasets/MCDD \
  --task detection \
  --num_classes 6 \
  --img_size 640 \
  --batch_size 16 \
  --epochs 300
```

### BDD (3 classes)
```bash
python train.py \
  --dataset_path datasets/BDD \
  --task detection \
  --num_classes 3 \
  --img_size 640
```

### MVTec-AD (Anomaly Segmentation)
```bash
python train.py \
  --dataset_path datasets/MVTec-AD/bottle \
  --task segmentation \
  --img_size 256
```

### Resume Training
```bash
python train.py \
  --dataset_path datasets/MCDD \
  --resume runs/train/last.pt
```

## Model Architecture

```
Input [B, 3, H, W]
  │
  ├── ConvModule (stride=2) ──→ [B, C, H/2, W/2]
  │
  ├── MDAM × 4 (Encoder)
  │     Stage 1: [B, C, H/2, W/2]   → [B, 2C, H/4, W/4]
  │     Stage 2: [B, 2C, H/4, W/4]  → [B, 4C, H/8, W/8]
  │     Stage 3: [B, 4C, H/8, W/8]  → [B, 8C, H/16, W/16]
  │     Stage 4: [B, 8C, H/16, W/16]→ [B, 16C, H/32, W/32]
  │
  ├── SSGCM (Bottleneck) ──→ [B, 16C, H/32, W/32]
  │
  ├── Decoder (SSGCM × 2 + GLFDM × 2)
  │     + Skip connections from encoder
  │
  ├── ConvModule ──→ [B, C, H/2, W/2]
  │
  └── Head
        Detection: (cls_out, reg_out)
        Segmentation: anomaly_map [B, 1, H/2, W/2]
```

## Configuration

Edit `configs/default.yaml` to customize:
- Model hyperparameters (base_dim, d_state, expand)
- Training settings (epochs, lr, optimizer)
- Loss weights
- Data augmentation
- Hardware settings

## Citation

```bibtex
@article{wu2026castmamba,
  title={CastMamba: Efficient Defect Detection in Mechanical Castings via State-Space Modeling and Multi-Scale Attention},
  author={Wu, HaoQiong and Li, Yuheng},
  journal={Scientific Reports},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
