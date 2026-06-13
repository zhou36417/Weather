# Weather Four-Class Classification

PyTorch/TorchVision project for four-class weather image classification:

- `sunny`
- `cloudy`
- `rainy`
- `snowy`

The repository contains training, evaluation, prediction, K-Fold, ensemble, and dataset preparation scripts. Large raw datasets, generated images, virtual environments, and model checkpoints are intentionally excluded from Git.

## Latest Result

Latest complete OOF ensemble result:

```text
runs/weather4_kfold_generalize_partial_oof/best_metrics.json
```

| Metric | Value |
|---|---:|
| Images | 16,413 |
| Ensemble | `convnext_tiny` + `efficientnet_b3` |
| Weights | `0.51 / 0.49` |
| Accuracy | `95.19%` |
| Macro F1 | `95.09%` |

Per-class F1:

| Class | F1 |
|---|---:|
| sunny | `95.73%` |
| cloudy | `94.86%` |
| rainy | `94.01%` |
| snowy | `95.77%` |

## Project Layout

```text
scripts/
  prepare_weathernet4.py
  prepare_weather4_from_dir.py
  split_weather4_dataset.py
  make_weather4_folds.py
  train_weather4.py
  evaluate_weather4.py
  predict_weather4.py
  run_weather4_kfold.py
  ensemble_weather4.py
  ensemble_weather4_kfold_oof.py
  weather4_lib.py

reports/
  weather_dataset_cleaning_summary.md
  weather_training_code_summary.md

data/processed/
  Lightweight CSV/JSON/Markdown metadata and split files.

runs/weather4_kfold_generalize_partial_oof/
  Lightweight OOF metrics and ensemble search outputs.
```

## Install

```bash
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build for GPU training when available.

## Data Notes

Large files are not tracked:

- raw parquet files in `data/external/raw/`
- extracted image folders such as `data/data/`
- generated image folders under `data/processed/**/images/`
- model checkpoints such as `*.pth`

Tracked processed CSV files contain image paths and labels. To reproduce training, place or regenerate the corresponding image files at the paths referenced by those CSVs.

## Train

Single-model training example:

```bash
python scripts/train_weather4.py \
  --project-root . \
  --train-csv data/processed/weather4_manual_clean_v2_partial_5fold/fold_0_train.csv \
  --val-csv data/processed/weather4_manual_clean_v2_partial_5fold/fold_0_val.csv \
  --label-map data/processed/weather4_from_dir_70_15_15/label_map.json \
  --output-dir runs/weather4_example/convnext_tiny/fold_0 \
  --model convnext_tiny \
  --weights imagenet \
  --epochs 15 \
  --batch-size 32 \
  --image-size 224 \
  --class-weights balanced \
  --label-smoothing 0.05 \
  --aug-preset generalize \
  --device auto
```

K-Fold training example:

```bash
python scripts/run_weather4_kfold.py \
  --project-root . \
  --fold-dir data/processed/weather4_manual_clean_v2_partial_5fold \
  --run-root runs/weather4_kfold_generalize_partial \
  --models convnext_tiny efficientnet_b3 \
  --folds 0 1 2 3 4 \
  --device auto \
  --lr 0.0001 \
  --label-smoothing 0.05 \
  --aug-preset generalize
```

## Evaluate / Ensemble

OOF ensemble outputs are stored under:

```text
runs/weather4_kfold_generalize_partial_oof/
```

Key files:

- `summary.json`
- `best_metrics.json`
- `best_confusion.csv`
- `model_oof_metrics.csv`
- `ensemble_search.csv`
- `oof_errors.csv`
- `oof_predictions.csv`

## Repository Policy

This repository is intended to track code, documentation, data manifests, split metadata, and lightweight metrics only. Keep raw datasets and checkpoints in external storage or Git LFS if they must be shared.
