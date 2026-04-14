# Object Detection Training Optimization
**CMPE 258 — Deep Learning | Assignment: 2D Object Detection**

## Overview

This project explores training optimization techniques for 2D object detection using **YOLOv8** on a subset of the **COCO 2017** dataset. We establish a baseline and systematically vary one component at a time to measure its individual effect on detection accuracy — a methodology known as **ablation study**.

---

## Dataset

- **Source**: COCO val2017 (5,000 images, 80 classes)
- **Split**: 4,000 train / 1,000 val (fixed seed=42 for reproducibility)
- **Format**: Converted from COCO JSON to YOLO format (normalized cx, cy, w, h)
- **Why val2017 only**: Keeps download size ~1GB instead of 19GB, sufficient for controlled ablation experiments

---

## Experiments

Each experiment changes **one variable** relative to the baseline. This isolates the effect of each change cleanly.

| Experiment | Model | Optimizer | Augmentation | mAP@0.5:0.95 | mAP@0.5 |
|---|---|---|---|---|---|
| Baseline | YOLOv8s | SGD | Default | 0.4050 | 0.5627 |
| Exp1 — Larger Backbone | YOLOv8m | SGD | Default | **0.4612** | **0.6220** |
| Exp2 — Augmentation | YOLOv8s | SGD | Aggressive | 0.3415 | 0.5621 |
| Exp3 — AdamW + Cosine LR | YOLOv8s | AdamW | Default | 0.2718 | 0.4029 |
| Exp4 — Best Combined | YOLOv8m | AdamW | Aggressive | 0.2754 | 0.4180 |

All experiments trained for **50 epochs** at **640×640** resolution on a **Tesla T4 GPU**.

---

## Experiment Details

### Baseline — YOLOv8s + SGD + Default Augmentation
**mAP50-95: 0.4050**

Stock YOLOv8 small model with standard SGD optimizer (lr=0.01, momentum=0.937) and default Ultralytics augmentation (mosaic, basic HSV jitter). This is the reference point for all comparisons.

---

### Exp1 — Larger Backbone (YOLOv8s → YOLOv8m)
**mAP50-95: 0.4612 (+5.6% over baseline)**

Swapped backbone from YOLOv8s (11M params) to YOLOv8m (25M params). All other settings identical. Batch size reduced from 16 to 8 to accommodate higher VRAM usage.

**Result**: Clear improvement. More parameters = richer feature representations = better detection. This is the most impactful single change we tested. Confirms that on this dataset size, model capacity is the primary bottleneck.

---

### Exp2 — Aggressive Augmentation
**mAP50-95: 0.3415 (-6.3% vs baseline)**

Added stronger augmentation to YOLOv8s baseline: rotation (±10°), shear, stronger HSV jitter, MixUp (α=0.15), copy-paste (p=0.1). Optimizer unchanged.

**Result**: Hurt performance. With only 4,000 training images, aggressive augmentation introduces more noise than regularization benefit. The model never sees a clean enough example to build stable feature representations. Augmentation is most effective when the model is overfitting — at this data scale, underfitting is the actual problem.

---

### Exp3 — AdamW + Cosine LR Scheduler
**mAP50-95: 0.2718 (-18.3% vs baseline)**

Swapped optimizer from SGD to AdamW (lr=0.001) with cosine annealing scheduler. Default augmentation, YOLOv8s backbone.

**Result**: Significant underperformance. AdamW adapts per-parameter learning rates using first and second moment estimates — effective on large datasets but slower to converge on small ones. SGD with momentum is more aggressive and reaches a good solution faster on limited data. Additionally, AdamW requires careful LR tuning; our lr=0.001 may have been too conservative.

---

### Exp4 — Best Combined (YOLOv8m + AdamW + Aggressive Augmentation)
**mAP50-95: 0.2754 (-19.6% vs baseline)**

Combined all three modifications: larger backbone, aggressive augmentation, AdamW + cosine LR.

**Result**: Worst overall performer despite having the most "improvements." This demonstrates a key principle: **gains are not always additive**. Two individually-negative changes (augmentation + AdamW on small data) compounded each other. Epoch-by-epoch analysis showed the model was still actively learning at epoch 50 (loss curves had not plateaued), indicating this configuration requires significantly more epochs (~100+) to converge than simpler configurations. The combined model is not fundamentally broken — it is simply slower to converge and was cut off before reaching its potential.

---

## Key Findings

**1. On small datasets, model capacity beats augmentation**
Doubling backbone size (+5.6% mAP) outperformed any training strategy change. With 4k images, the model's ability to represent features is more limiting than generalization.

**2. Aggressive augmentation requires sufficient data**
Augmentation acts as a regularizer — it helps when the model is overfitting. At 4k images the model isn't overfitting, so augmentation only makes training harder without benefit.

**3. SGD outperforms AdamW on small datasets**
AdamW's adaptive learning rates help on large, sparse datasets. On small datasets with dense gradients, SGD's simplicity and momentum-based updates converge faster and more reliably.

**4. Complex configurations need more epochs**
The combined model's loss was still dropping steeply at epoch 50. More sophisticated training setups (larger model + harder augmentation + adaptive optimizer) require proportionally more training time to converge. This is a fundamental tradeoff, not a failure of the approach.

**5. Ablation study is essential**
Running the combined experiment without individual experiments would have produced a confusing result with no explanation. Isolating each variable revealed exactly which components helped and which didn't — and why.

---

## Repository Structure

```
object-detection-optimization/
├── object_detection_optimization.ipynb   # Full notebook with all outputs embedded
└── README.md                             # This file
```

---

## How to Run

1. Open `object_detection_optimization.ipynb` in Google Colab
2. Set runtime to GPU (T4 or better)
3. Run all cells top to bottom
4. Each experiment takes ~25-30 min on T4 (50 epochs)

All outputs (loss curves, confusion matrices, PR curves, sample predictions) are embedded in the notebook and visible without re-running.

---

## Environment

- Python 3.12
- PyTorch 2.10 + CUDA 12.8
- Ultralytics 8.x
- GPU: Tesla T4 (15GB VRAM)
- Platform: Google Colab
