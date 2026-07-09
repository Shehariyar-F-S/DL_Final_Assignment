# IDL26 — Operation Cyber-Histology

**Post-Incident Pipeline Reconstruction and ML Engineering**

<br>

**Authors:**

| Name | Enrollment Number |
| :--- | :--- |
| Lavanya Nagaraj | 10001071 |
| Shehariyar Firdous Shaikh | 10000753 |

---

**Course:** MAI/IDL SS26 — Final Assignment

**Date:** 09 July 2026

---

## Overview
This repository contains the fully restored, production-grade multi-class clinical image classification pipeline for BioHealth Diagnostics Global, reconstructed across three phases:
*   **Task 1** — Pipeline reconstruction and bug fixing
*   **Task 2** — Green Initiative: lightweight models and efficiency profiling
*   **Task 3** — Transfer learning for scarce organs dataset

## Branch Structure
```text
main                      <- Final merged solution
task1-bug-fixes           <- Task 1 development history
task2-green-initiative    <- Task 2 development history
task3-organs-transfer     <- Task 3 development branch
messy-draft-backup        <- Original development history (preserved)
```

## Repository Structure
```text
DL_Final_Assignment/
├── Code/
│   ├── data.py                 # Data loading, preprocessing and augmentation
│   ├── models.py               # All architectures: Task 1, Task 2 Green, Task 3 Transfer
│   ├── fit.py                  # Trainer class with profiling
│   ├── train.py                # Task 1 and 2 pipeline orchestrator
│   ├── train_task3.py          # Task 3 transfer learning benchmark
│   └── config.json             # Central configuration file
├── Data/                       # Dataset files (not tracked by git)
├── AUDIT_LOG.md
├── REPORT.md
└── README.md
```

## Prerequisites
*   Python 3.10+
*   PyTorch 2.0+ with CUDA (optional)
*   scikit-learn, matplotlib, torchvision

## Installation
```bash
git clone https://github.com/Shehariyar-F-S/DL_Final_Assignment.git
cd DL_Final_Assignment
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision scikit-learn matplotlib
```
Place dataset files in `Data/` folder: `cells.pt`, `chest.pt`, `lesions.pt`, `orgs.pt`, `organs.pt`

## Configuration
All parameters controlled via `Code/config.json`:
```json
{
 "DATA_PATH": "../Data",
 "BATCH_SIZE": 64,
 "LEARNING_RATE": 0.001,
 "EPOCHS": 20,
 "VAL_SPLIT": 0.1,
 "PRUNE_AMOUNT": 0.3,
 "FINETUNE_EPOCHS": 5,
 "MODELS": ["ResNet18", "VGG16", "AlexNet", "GreenResNet18", "GreenVGG16"],
 "DATASETS": [
   {"DATA": "chest", "CHANNELS": 1, "NUM_CLASSES": 2},
   {"DATA": "lesions", "CHANNELS": 3, "NUM_CLASSES": 7},
   {"DATA": "orgs", "CHANNELS": 1, "NUM_CLASSES": 11},
   {"DATA": "cells", "CHANNELS": 3, "NUM_CLASSES": 8},
   {"DATA": "organs", "CHANNELS": 1, "NUM_CLASSES": 11}
 ]
}
```
*Important: `orgs` must appear before `organs` in DATASETS. Task 3 loads weights saved from orgs training.*

## Execution

### Task 1 and 2
```bash
cd Code
python train.py
```

### Task 3 — Transfer Learning
```bash
cd Code
python train_task3.py
```

## Models

### Task 1 — Core Architectures
| Model | Approx. Params | Key Feature |
| :--- | :--- | :--- |
| AlexNet | ~61M* | Decreasing kernel sizes 7 to 5 to 3 |
| VGG16 | ~138M* | Systematic 3x3 convolutions |
| ResNet18 | ~11.2M | Skip connections |

*\*Adapted for 64x64 inputs — fewer params than original.*

### Task 2 — Green Initiative
| Model | Total Params | Active After 30% Pruning | Reduction |
| :--- | :--- | :--- | :--- |
| ResNet18 | ~11.2M | ~11.2M | baseline |
| GreenResNet18 | ~2.7M | ~1.9M active | ~83% fewer than ResNet18 |
| GreenVGG16 | ~3.1M | <2.5M active | ~98% fewer than VGG16 |

*Approximate values. Two-stage strategy: (1) channel halving, (2) L1 unstructured pruning (30%). Exact count: `sum(p.numel() for p in model.parameters())`*

### Task 3 — Transfer Learning
| Model | Strategy | Approx. Trainable Params |
| :--- | :--- | :--- |
| FrozenTransferResNet18 | All conv frozen, classifier only | ~2K |
| FineTunedTransferResNet18 | Stage 4 and classifier unfrozen | ~2.08M |

## Key Engineering Decisions
*   Config-driven pipeline — all hyperparameters in `config.json`
*   Early stopping — patience=7 epochs
*   Best model checkpointing — `copy.deepcopy(state_dict())`
*   Data augmentation — training data only
*   LR scheduling — `ReduceLROnPlateau`
*   Macro-averaged metrics — fair for imbalanced datasets
*   Domain-specific transfer — `orgs` weights used for `organs` (same modality)
*   L1 unstructured pruning — 30% weight sparsity with fine-tuning recovery

## Task 1 Benchmark Results
| Model | cells >=90% | chest >=87% | lesions >=67% | orgs >=83% |
| :--- | :--- | :--- | :--- | :--- |
| ResNet18 | 96.84% | 91.19% | 68.68% | 92.27% |
| VGG16 | 97.54% | 92.95% | 77.86% | 93.16% |
| AlexNet | 94.62% | 87.66% | 76.96% | 90.35% |

All targets met. Full analysis in `REPORT.md`.
Bug fixes and Audits in `AUDIT_LOG.md` 


