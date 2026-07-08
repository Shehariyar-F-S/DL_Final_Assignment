# AUDIT_LOG.md — Operation Cyber-Histology
## BioHealth Diagnostics Global — Post-Incident Pipeline Reconstruction
### MAI/IDL SS26 — Bug, Corruption & Anti-Pattern Report

---

## Section 1 — Critical Bug Fixes

| # | File | Manifestation | Mathematical / Logical Root Cause | Structural Correction | Commit Hash |
|---|---|---|---|---|---|
| 1 | `data.py` | `FileNotFoundError` at runtime — pipeline cannot load any dataset | Filename suffix `_data.pt` does not match actual filenames on disk (e.g. `cells.pt`). `Path(data_path) / f"{data}_data.pt"` resolves to a non-existent path | Removed `_data` suffix: `f"{data}_data.pt"` → `f"{data}.pt"` | `7577bff` |
| 2 | `data.py` | Silent — model trains and validates on overlapping data, inflating validation metrics | `train_data = data_dict['train_images']` assigned the **full** training set without slicing. Validation samples included in training causes data leakage: $D_{train} \cap D_{val} \neq \emptyset$ | Applied correct slicing: `train_data = shuffled_images[:val_start]` and `val_data = shuffled_images[val_start:]` | `4794e20`, `24f2240` |
| 3 | `data.py` | `RuntimeError` — `CrossEntropyLoss` receives labels of wrong shape | Labels stored as 2D tensor $\mathbf{y} \in \mathbb{R}^{N \times 1}$ but `CrossEntropyLoss` requires 1D integer class indices $\mathbf{y} \in \mathbb{Z}^{N}$ | Applied `.squeeze().long()` inside `preprocess()` to convert `[N, 1]` → `[N]` | `fe62003` |
| 4 | `fit.py` | Numerical — gradients explode or oscillate; training loss diverges across batches | Missing `optimizer.zero_grad()` causes gradients to accumulate: $g_t^{accum} = \sum_{i=0}^{t} g_i$ instead of $g_t$ only, producing oversized parameter updates | Added `self.optimizer.zero_grad()` at start of each batch iteration in `train_one_epoch()` | `3b76e5a` |
| 5 | `fit.py` | `UnboundLocalError` — variable `sum` cannot be accessed as local variable | Variable named `sum` shadows Python built-in `sum()`. Python scoping treats it as a local variable referenced before assignment | Renamed `sum` → `total` throughout `train_one_epoch()` | `bc44193` |
| 6 | `fit.py` | `NameError` — `all_labels` and `all_preds` referenced before assignment in `evaluate()` | Lists used to accumulate predictions were never initialised before the evaluation loop | Initialised before loop: `all_labels, all_preds = [], []` | `d6242f1` |
| 7 | `models.py` | `TypeError` — `ResNet18.forward()` returns `None` to loss function | Missing `return` statement causes `forward()` to implicitly return `None`. `criterion(None, labels)` raises `TypeError` | Added `return self.classifier(out)` at end of `ResNet18.forward()` | `457e4e3` |
| 8 | `models.py` | `RuntimeError` — shape mismatch in `VGGBlock` convolutions | `current_in_channels` never updated inside loop. Each `Conv2d` received original `in_channels` instead of previous layer's `out_channels`: expected $C_{out}^{l-1}$ but received $C_{in}^{0}$ | Added `current_in_channels = out_channels` at end of each loop iteration | `40e9a02` |
| 9 | `models.py` | Silent — `ResNet18` behaves as linear classifier regardless of depth | `activation_str = None` causes `getattr(nn, None)` to fail. Without non-linearity: $f(x) = W_n \cdots W_1 x$, collapsing the deep network to a single linear transformation | Set `activation_str = "ReLU"` and validated via `getattr(nn, activation_str)` | `3903154` |
| 10 | `models.py` | `TypeError` — `AlexNet.__init__()` missing required arguments `in_channels` and `num_classes` | Constructor did not declare `in_channels` and `num_classes`. Input channels hardcoded to `3` and output classes to `11`, making model incompatible with grayscale datasets and variable class counts | Added `in_channels` and `num_classes` to `AlexNet.__init__(self, in_channels, num_classes, **kwargs)` | `d33cc0f` |
| 11 | `models.py` | `RuntimeError` — shape mismatch between `AlexNet` feature extractor output and classifier input | Classifier's first `Linear(2048)` received $192 \times H \times W \neq 2048$ without fixed spatial pooling. Spatial dims vary with input resolution | Added `self.avgpool = nn.AdaptiveAvgPool2d((2, 2))` fixing output to $(192, 2, 2)$. Updated: `nn.Linear(192 * 2 * 2, 1024)` | `3d2b39c`, `89ee027` |
| 12 | `models.py` | `AttributeError` — `VGG16.forward()` calls `self.avgpool` which was never defined | `self.avgpool` referenced in `forward()` without instantiation in `__init__()` | Added `self.avgpool = nn.AdaptiveAvgPool2d((2, 2))` to `VGG16.__init__()` | `f881080` |
| 13 | `models.py` | Silent — `AlexNet` output hardcoded to `11` classes, incompatible with other datasets | Final `Linear` layer hardcoded: `nn.Linear(1024, 11)`. Produces wrong logit dimensions for `cells=8`, `chest=2`, `lesions=7`, `orgs=11` | Changed to `nn.Linear(1024, num_classes)` | `c6ed3a6` |
| 14 | `train.py` | `FileNotFoundError` — entire pipeline cannot start | No `config.json` existed — configuration infrastructure wiped by Dr. Vance | Created `config.json` from scratch with: `MODELS`, `DATASETS`, `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `VAL_SPLIT`, `DATA_PATH` | `818d8ec` |
| 15 | `train.py` | Silent — only last dataset trains; all others silently overwritten | `train_loader`, `model`, `optimizer` initialised outside inner dataset loop. Each iteration overwrites previous values | Moved all dataset-dependent operations inside `for dataset_config` loop with correct indentation | `f657cf6` |
| 16 | `train.py` | Silent — model fails to learn; accuracy never improves beyond chance | `drop_rate=0.99` zeroes $99\%$ of neurons: $\mathbb{E}[\text{active}] = 0.01n \approx 0$. Network receives no signal and cannot update weights | Removed hardcoded `drop_rate=0.99`. Model uses default `kwargs.get("drop_rate", 0.5)` | `6d1b6d3` |
| 17 | `train.py` | `NameError` — `test_loader` not correctly unpacked from `get_loaders()` | `get_loaders()` returns three loaders but `test_loader` was not correctly received | Fixed: `train_loader, val_loader, test_loader = get_loaders(...)` | `56ec30c` |
| 18 | `models.py` | Silent — `ResNet18` missing dropout after removing broken `drop_rate=0.99` | No regularisation in classifier increases overfitting risk on smaller datasets | Added `drop_rate = kwargs.get("drop_rate", 0.5)` and `nn.Dropout(p=drop_rate)` to `ResNet18.classifier` | `ae0775d` |

---

## Section 2 — Production-Grade Enhancements

| # | File | Enhancement | Justification | Commit Hash |
|---|---|---|---|---|
| 1 | `data.py` | Added `torch.manual_seed(42)` + `torch.randperm()` for reproducible shuffling | Ensures uniform class distribution across splits and reproducible results | `f565b5c`, `34b4b13` |
| 2 | `data.py` | Added `map_location=torch.device('cpu')` to `torch.load()` | Prevents device mismatch errors across CPU and GPU environments | `7092d16` |
| 3 | `data.py` | Added `preprocess()` normalising images to `[0,1]` and converting labels to `long` | Raw pixel values `[0,255]` cause gradient instability. Normalisation ensures stable backpropagation | `ad053f7`, `fe62003` |
| 4 | `data.py` | Added `AugmentedDataset` with random flips, rotation and affine transforms | Artificially expands training diversity, improving generalisation on smaller datasets | `fd2886d` |
| 5 | `fit.py` | Added `precision_recall_fscore_support` for macro-averaged evaluation metrics | Accuracy alone is misleading for imbalanced medical datasets. Macro F1 weights all classes equally | `306193d` |
| 6 | `fit.py` | Added loss curve tracking and `matplotlib` plotting per dataset | Visual inspection of train vs validation loss essential for diagnosing overfitting | `ee4f962` |
| 7 | `fit.py` | Added best model weight checkpointing via `model.state_dict()` | Saves peak validation weights, preventing degraded overfit final model | `c435186` |
| 8 | `fit.py` | Added early stopping with configurable `patience` parameter | Prevents unnecessary computation when validation loss stagnates | `759d47c`, `0dba4da` |
| 9 | `fit.py` | Added `dataset_name` parameter to `fit()` for labelled loss curve saving | Enables per-dataset loss curves for benchmark comparison | `c9f803f` |
| 10 | `train.py` | Added configuration-driven model loading via `getattr(models, model_name)` | All three models run via single `config.json` — satisfies modularity requirement | `bd1cd98` |
| 11 | `train.py` | Added detailed print logs for all evaluation metrics | Clear visibility of Accuracy, Precision, Recall, F1 per model-dataset combination | `a45097f` |
| 12 | `config.json` | Created complete configuration file from scratch | Centralises all hyperparameters. Enables full pipeline execution without manual code edits | `818d8ec`, `fc0fd07` |

---

## Summary

| Category | Count |
|---|---|
| Critical Bug Fixes | 18 |
| Production-Grade Enhancements | 12 |
| **Total Entries** | **30** |

---

*Audit conducted by: Shehariyar Firdous Shaikh and Lavanya Nagaraj*
*Date: July 2026*
*Course: MAI/IDL SS26 — Deep Learning*
*Assignment: Operation Cyber-Histology*
