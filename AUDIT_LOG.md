# Incident Audit Log — Operation Cyber-Histology
## BioHealth Diagnostics Global — Post-Incident Pipeline Reconstruction

---

## Section 1 — Bug Fixes (Critical)

| # | File | Manifestation | Mathematical / Logical Root Cause | Structural Correction | Commit Hash |
|---|---|---|---|---|---|
| 1 | `data.py` | `FileNotFoundError` at runtime — pipeline cannot load any dataset | Filename suffix `_data.pt` does not match actual filenames on disk (e.g. `cells.pt`) | Removed `_data` suffix: `f"{data}_data.pt"` → `f"{data}.pt"` | `7577bff` |
| 2 | `data.py` | Silent — model trains and validates on overlapping data, inflating validation accuracy | `train_data` was assigned the full training set `data_dict['train_images']` without slicing, causing the validation samples to be included in training. Formally: $D_{train} \cap D_{val} \neq \emptyset$ | Applied correct slicing: `train_data = shuffled_images[:val_start]`, `val_data = shuffled_images[val_start:]` | `4794e20`, `24f2240` |
| 3 | `data.py` | `RuntimeError` — CrossEntropyLoss receives labels of shape `[N, 1]` instead of expected `[N]` | Labels stored as 2D tensor $\mathbf{y} \in \mathbb{R}^{N \times 1}$ but `CrossEntropyLoss` requires 1D integer class indices $\mathbf{y} \in \mathbb{Z}^{N}$ | Applied `.squeeze().long()` inside preprocessing to convert `[N, 1]` → `[N]` | `fe62003` |
| 4 | `fit.py` | Gradient explosion — loss diverges or oscillates wildly across batches | Missing `optimizer.zero_grad()` causes gradients to **accumulate** across batches. Formally: $g_t = g_t + g_{t-1} + ... + g_0$ instead of $g_t$ only | Added `self.optimizer.zero_grad()` before forward pass in `train_one_epoch()` | `3b76e5a` |
| 5 | `fit.py` | `UnboundLocalError` — `sum` cannot be accessed as local variable | Variable named `sum` shadows Python's built-in `sum()` function, causing `UnboundLocalError` when Python's scoping rules attempt to resolve the name | Renamed variable from `sum` to `total` throughout `train_one_epoch()` | `bc44193` |
| 6 | `fit.py` | Silent — `all_labels` and `all_preds` referenced before assignment in `evaluate()` | Lists `all_labels` and `all_preds` used to accumulate predictions were never initialised before the evaluation loop, causing `NameError` or silent corruption | Initialised both lists before loop: `all_labels, all_preds = [], []` | `d6242f1` |
| 7 | `models.py` | `AttributeError` — ResNet18 forward pass fails silently | Missing `return` statement in `ResNet18.forward()` causes the method to return `None` instead of the output tensor, breaking the computational graph | Added explicit `return self.classifier(out)` at end of `forward()` | `457e4e3` |
| 8 | `models.py` | `RuntimeError` — shape mismatch in VGGBlock convolutions | `current_in_channels` was never updated inside the VGGBlock loop, causing each subsequent convolution to receive wrong number of input channels | Added `current_in_channels = out_channels` at end of each loop iteration | `40e9a02` |
| 9 | `models.py` | Silent — ResNet18 uses no activation function (`None`) causing dead network | `activation_str` was set to `None` or invalid value, causing `getattr(nn, None)` to fail or return identity, eliminating all non-linearity from the network. Without activation: $f(x) = Wx$ — a purely linear model regardless of depth | Initialised `activation_str = "ReLU"` and validated via `getattr(nn, activation_str)` | `3903154` |
| 10 | `train.py` | `FileNotFoundError` — pipeline cannot find `config.json` | No `config.json` file existed in the repository — entire configuration infrastructure was wiped by Dr. Vance | Created `config.json` from scratch with all required fields: `MODELS`, `DATASETS`, `BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `VAL_SPLIT`, `DATA_PATH` | `818d8ec` |
| 11 | `train.py` | Silent — only last dataset trains correctly, all others ignored | `train_loader`, `model`, `criterion`, `optimizer` were defined outside the dataset loop, causing only the final dataset's values to persist for training | Moved all dataset-dependent operations inside the inner `for dataset_config` loop with correct indentation | `f657cf6` |
| 12 | `train.py` | Silent — `drop_rate=0.99` causes extreme regularisation, killing all learned features | Dropout probability of `0.99` randomly zeroes 99% of neurons, making it mathematically impossible for the network to learn. Formally: $\mathbb{E}[\text{active neurons}] = 0.01 \times n \approx 0$ | Reset dropout to default `0.5` and initialised model with correct `in_channels` and `num_classes` from dataset config | `6d1b6d3` |
| 13 | `train.py` | `SyntaxError` / `NameError` — `test_loader` referenced incorrectly | Variable `test_loader` was either missing or incorrectly named, preventing final evaluation from executing | Fixed variable assignment to correctly unpack `train_loader, val_loader, test_loader = get_loaders(...)` | `bd1cd98` |

---

## Section 2 — Enhancements (Production-Grade Improvements)

| # | File | Enhancement | Justification | Commit Hash |
|---|---|---|---|---|
| 1 | `data.py` | Added `torch.manual_seed(42)` and `torch.randperm()` for reproducible shuffling before train/val split | Ensures class distribution is uniform across splits and results are reproducible across runs | `f565b5c`, `34b4b13` |
| 2 | `data.py` | Added `map_location=torch.device('cpu')` to `torch.load()` | Ensures data always loads on CPU regardless of environment, preventing device mismatch errors on GPU servers | `7092d16` |
| 3 | `data.py` | Added `preprocess()` function normalising images to `[0, 1]` and converting labels to `long` | Raw pixel values `[0, 255]` cause gradient instability. Normalisation ensures $x \in [0,1]$ keeping gradients well-behaved during backpropagation | `ad053f7`, `fe62003` |
| 4 | `fit.py` | Added `precision_recall_fscore_support` from `sklearn.metrics` | Accuracy alone is insufficient for imbalanced medical datasets. Precision, Recall and F1-score provide a complete picture of model performance per class | `306193d` |
| 5 | `fit.py` | Added loss curve plotting via `matplotlib` | Visual inspection of training vs validation loss curves is essential for detecting overfitting and diagnosing training stability | `ee4f962` |
| 6 | `fit.py` | Added best model weight tracking and checkpointing via `torch.save()` | Saves the model at its peak validation performance, preventing the final weights from being a degraded overfit version | `c435186` |
| 7 | `train.py` | Added `dataset_name` parameter to `fit()` for logging | Enables loss curves and logs to be labelled per dataset, essential for multi-dataset benchmark comparison | `4a7cba9` |
| 8 | `train.py` | Added detailed print logs for all evaluation metrics | Provides clear visibility of Test Accuracy, Precision, Recall and F1-Score per model-dataset permutation | `a45097f` |

---

## Summary Statistics

| Category | Count |
|---|---|
| Critical Bug Fixes | 13 |
| Production Enhancements | 8 |
| **Total Entries** | **21** |

---