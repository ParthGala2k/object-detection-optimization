# Object Detection Training Optimization
**CMPE 258 - Deep Learning | Assignment: 2D Object Detection**

---

## Assignment Requirements Checklist (Option 1: Training Optimization)

| Requirement | Status |
|---|---|
| Baseline model setup | Done - YOLOv8s, SGD, default augmentation |
| Modified/improved training approach | Done - 4 experiments across backbone, augmentation, optimizer |
| Description of what changed, why, and how it affects performance | Done - per-experiment analysis below |
| COCO-style evaluation results before and after changes | Done - mAP@0.5:0.95 and mAP@0.5 for all experiments |
| Architecture modification (backbone) | Done - YOLOv8s to YOLOv8m |
| Training strategy change (augmentation) | Done - aggressive augmentation pipeline |
| Training strategy change (optimizer + scheduler) | Done - SGD to AdamW with cosine annealing |

---

## Overview

This project systematically explores training optimization for 2D object detection using YOLOv8 on a COCO 2017 subset. The methodology is an **ablation study**: change exactly one variable per experiment, measure the effect, and use the findings to reason about the next experiment. This isolates causal relationships rather than chasing correlated improvements.

---

## Dataset

| Property | Value |
|---|---|
| Source | COCO val2017 |
| Total images | 5,000 (80 classes) |
| Train split | 4,000 images |
| Val split | 1,000 images |
| Split seed | 42 (fixed for reproducibility) |
| Annotation format | Converted from COCO JSON to YOLO format (normalized cx, cy, w, h) |

COCO val2017 was chosen over train2017 to keep the download size at ~1GB instead of 19GB, while still providing a standard benchmark for evaluation.

---

## Results Summary

All experiments trained for **50 epochs** at **640x640** resolution on a **Tesla T4 GPU**.

| Experiment | Model | Optimizer | Augmentation | mAP@0.5:0.95 | mAP@0.5 | Delta vs Baseline |
|---|---|---|---|---|---|---|
| Baseline | YOLOv8s | SGD | Default | 0.4050 | 0.5627 | - |
| Exp 1: Larger Backbone | YOLOv8m | SGD | Default | **0.4612** | **0.6220** | +5.6% |
| Exp 2: Augmentation | YOLOv8s | SGD | Aggressive | 0.3415 | 0.5621 | -6.3% |
| Exp 3: AdamW + Cosine LR | YOLOv8s | AdamW | Default | 0.2718 | 0.4029 | -18.3% |
| Exp 4: Best Combined | YOLOv8m | AdamW | Aggressive | 0.2754 | 0.4180 | -19.6% |

---

## Experiments

### Baseline: YOLOv8s + SGD + Default Augmentation

**mAP@0.5:0.95: 0.4050 | mAP@0.5: 0.5627**

Stock YOLOv8 small model (11M parameters) trained with SGD (lr=0.01, momentum=0.937, weight decay=0.0005) and default Ultralytics augmentation (mosaic, basic HSV color jitter). No modifications.

This is the reference point for all comparisons. Every subsequent experiment changes exactly one variable so the effect can be attributed clearly.

---

### Exp 1: Larger Backbone (YOLOv8s to YOLOv8m)

**mAP@0.5:0.95: 0.4612 | mAP@0.5: 0.6220 | +5.6% over baseline**

**What changed:** Swapped backbone from YOLOv8s (11M params) to YOLOv8m (25M params). All other settings identical. Batch size reduced 16 to 8 to fit within T4 VRAM.

**Why:** More parameters allow the model to learn richer, more discriminative feature representations. The hypothesis was that with only 4,000 training images, the small model was capacity-limited rather than data-limited.

**Result:** Best performing experiment. A larger backbone directly improved the model's ability to extract features from the same data. This confirms that at this dataset scale, model capacity is the primary bottleneck. The takeaway going into the next experiment: the architecture matters more than training tricks when data is limited.

---

### Exp 2: Aggressive Augmentation

**mAP@0.5:0.95: 0.3415 | mAP@0.5: 0.5621 | -6.3% vs baseline**

**What changed:** Applied stronger augmentation to the YOLOv8s baseline: rotation (10 degrees), shear (2 degrees), stronger HSV color jitter, MixUp (alpha=0.15), and copy-paste augmentation (p=0.1). Backbone and optimizer unchanged.

**Why:** Augmentation acts as a regularizer by forcing the model to learn invariant features rather than memorizing pixel patterns. The hypothesis was that a more diverse training distribution would improve generalization on val.

**Result:** Hurt performance. With only 4,000 training images, the model is not overfitting in the first place. Aggressive augmentation makes each training example harder to learn from, causing underfitting instead of reducing overfitting. The model needs to see clean examples long enough to learn stable features before augmentation adds value. The key insight: augmentation helps most when the training dataset is large enough that the model starts to memorize it.

---

### Exp 3: AdamW Optimizer + Cosine LR Scheduler

**mAP@0.5:0.95: 0.2718 | mAP@0.5: 0.4029 | -18.3% vs baseline**

**What changed:** Swapped optimizer from SGD to AdamW (lr=0.001) with cosine annealing (final LR = lr * 0.01). Default augmentation and YOLOv8s backbone kept the same.

**Why:** AdamW maintains per-parameter adaptive learning rates using first and second moment estimates of gradients, which theoretically handles varied gradient scales better than SGD. Cosine annealing smoothly decays LR following a cosine curve, avoiding abrupt step drops.

**Result:** Significant underperformance. On small datasets with relatively dense, consistent gradients, SGD with momentum is more aggressive and converges faster. AdamW's adaptive rates are most beneficial when gradient distributions are sparse and varied, which requires larger datasets. The lr=0.001 also likely contributed, as AdamW is sensitive to LR scaling relative to batch size. This experiment motivated the question: what happens when we combine all three changes together?

---

### Exp 4: Best Combined (YOLOv8m + AdamW + Aggressive Augmentation)

**mAP@0.5:0.95: 0.2754 | mAP@0.5: 0.4180 | -19.6% vs baseline**

**What changed:** Combined all three modifications: YOLOv8m backbone, aggressive augmentation, AdamW with cosine LR.

**Why:** Based on Exp 1 showing backbone helps, the hypothesis was that combining all three might be additive, with the larger backbone compensating for the slower convergence of AdamW and the noise from augmentation.

**Result:** Worst overall performer despite having the most modifications. This demonstrates that improvements are not always additive. Epoch-by-epoch analysis revealed that all three losses (box, cls, dfl) were still declining steeply at epoch 50 with no sign of plateauing, and mAP@0.5:0.95 had progressed from 0.075 at epoch 1 to 0.230 by epoch 20 and 0.275 at epoch 50 - still clearly converging. The combined configuration is not fundamentally broken. It requires significantly more epochs (~100+) to converge because the larger model, harder augmentation, and adaptive optimizer each individually slow convergence, and their effects compound. The 50-epoch budget that was sufficient for simpler configurations was insufficient here.

---

## Key Findings

**1. Model capacity beats training tricks on small datasets**
Scaling up the backbone (+5.6% mAP) was the only change that consistently helped. With limited data, the model's representational capacity is the bottleneck.

**2. Augmentation requires a minimum dataset size to help**
Augmentation is a regularizer. It only helps when the model is overfitting. At 4,000 images the model was underfitting, so augmentation made training harder without any generalization benefit.

**3. SGD converges faster than AdamW on small datasets**
AdamW's per-parameter adaptive learning rates are most valuable on large, sparse datasets. On small datasets with consistent gradients, SGD with momentum is more direct and converges faster.

**4. More complex configurations need more epochs**
Every additional modification slows convergence. The combined model was still visibly learning at epoch 50. The training budget must scale with configuration complexity.

**5. Ablation study reveals what actually matters**
Running all experiments individually made it possible to explain the combined result. Without per-experiment isolation, the worst result would have been unexplainable.

---

## Repository Structure

```
object-detection-optimization/
├── object_detection_optimization.ipynb   # Full notebook with embedded outputs
└── README.md
```

---

## How to Run

1. Open `object_detection_optimization.ipynb` in Google Colab
2. Set runtime to GPU (T4 or better)
3. Run all cells top to bottom
4. Each experiment runs for 50 epochs (~25-30 min per experiment on T4)

All outputs including loss curves, confusion matrices, PR curves, and sample predictions are embedded in the notebook and visible without re-running.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10 + CUDA 12.8 |
| Ultralytics | 8.x |
| GPU | Tesla T4 (15GB VRAM) |
| Platform | Google Colab |
