# AUDIT LOG — Operation Cyber-Histology

## data.py

| # | File | Manifestation | Root Cause | Fix | Commit Hash |
|---|---|---|---|---|---|
| 1 | data.py | FileNotFoundError at runtime | Wrong filename suffix `_data.pt` | Changed to `{data}.pt` | `426482a` |
| 2 | data.py | Silent data leakage | Train set not sliced before val split | Applied `[:val_start]` slice | `8f2625e` |
| 3 | data.py | Biased validation split | Data not shuffled before splitting | Added `torch.randperm` shuffle and manual_seed(42) for reproducibility | `b48c528` |
| 4 | data.py | Unstable/slow training | Raw pixel values `[0-255]` not normalized | Divided by `255.0` | `44da953` |


## fit.py

| # | File | Manifestation | Root Cause | Fix | Commit Hash |
|---|---|---|---|---|---|
| 1 | fit.py |
| 2 | fit.py | 
| 3 | fit.py |
| 4 | fit.py |

## train.py

| # | File | Manifestation | Root Cause | Fix | Commit Hash |
|---|---|---|---|---|---|
| 1 | fit.py |
| 2 | fit.py | 
| 3 | fit.py |
| 4 | fit.py |

## models.py

| # | File | Manifestation | Root Cause | Fix | Commit Hash |
|---|---|---|---|---|---|
| 1 | fit.py |
| 2 | fit.py | 
| 3 | fit.py |
| 4 | fit.py |
