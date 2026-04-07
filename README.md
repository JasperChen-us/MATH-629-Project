# MATH-629 Project

2026 WN MATH 629 Final Project by Taoyi Chen and Fabian Shen.

This repository currently focuses on a **BTC 2023 limit order book pipeline** and a Deep Learning notebook. The main goal is to forecast short-horizon mid-price movement from order book snapshots, then compare several model families on the same time-split pipeline.

## Current Main Entry Point

The main training notebook is now:

- `main.ipynb`

This notebook is the primary place to:

- load the BTC 2023 cleaned bundle,
- build standardized FI-2010-like sequences,
- train a DeepLOB baseline,
- train a pure transformer baseline,
- train a Conv-Transformer hybrid,
- train naive and improved direction classifiers,
- compare validation and test behavior.

## Important Date Note

The raw file is named `BTC20230120.csv`, and the timestamps span:

- from `2023-01-09 22:17:40 UTC`
- to `2023-01-20 18:10:48 UTC`

## BTC 2023 Workflow

The current BTC 2023 workflow is:

1. Start from `data/Tardis_sample_data/BTC20230120.csv`.
2. Clean and normalize it with `clean_tardis_lob_dataset.py`.
3. Save the cleaned 1-second bundle to `data/BTC20230120_clean_1s.pkl`.
4. Inspect the file and compare available BTC samples in `sample_adjustment.ipynb`.
5. Train models in `main.ipynb`.

## BTC 2023 Files

This section lists only the files that are directly relevant to the current BTC 2023 data collection and adjustment workflow.

### `data/Tardis_sample_data/BTC20230120.csv`

This is the raw BTC 2023 order book file used by the current project workflow.

What it contains:

- millisecond timestamps,
- a flat FI-2010-style export layout,
- 10 bid levels and 10 ask levels,
- price and size information for each level.

Why it matters:

- it is the source dataset for the current experiments,
- it spans far more usable 1-second timestamps than the older BTC sample,
- it is the dataset the main training notebook now uses by default after cleaning.

### `clean_tardis_lob_dataset.py`

This is the main cleaning and conversion script for the BTC 2023 pipeline.

What it does:

- detects the input schema automatically,
- supports both raw Tardis-style CSVs and flat FI-2010-style exports,
- converts timestamps to UTC,
- deduplicates repeated timestamps,
- buckets snapshots to `1ms` or `1s`,
- keeps the **last snapshot per bucket**,
- builds a cleaned dataframe,
- builds a 40-feature FI-2010 matrix,
- stores the result as a pickle bundle.

Current default behavior:

- default input: `data/Tardis_sample_data/BTC20230120.csv`
- default output: `data/BTC20230120_clean_1s.pkl`

Why it matters:

- it is the script that turns raw BTC 2023 order book data into the format used everywhere else in the project,
- it makes the data directly usable by the PyTorch notebook,
- it stores metadata such as the date range, source format, chosen frequency, and feature columns.

### `sample_adjustment.ipynb`

This notebook is the BTC 2023 data inspection and adjustment notebook.

What it does:

- scans the available CSVs in `data/Tardis_sample_data`,
- defaults to `BTC20230120.csv`,
- profiles timestamp density,
- compares raw rows, unique milliseconds, and usable 1-second rows,
- previews the raw CSV layout,
- previews the normalized book layout,
- previews the cleaned 1-second dataframe.

Why it matters:

- it is the quickest way to verify that the BTC 2023 data is clean enough before training,
- it shows whether the selected file produces more usable data,
- it documents the choice to use the BTC 2023 sample rather than the older sample.

### `data/BTC20230120_clean_1s.pkl`

This is the cleaned BTC 2023 dataset used by the current training notebook.

What it stores:

- `clean_lob`: cleaned 1-second snapshots in dataframe form,
- `fi2010_features`: 40 FI-2010-style features in tabular form,
- `fi2010_matrix`: transposed feature matrix,
- `metadata`: source file, format, date range, feature names, and timestamp profile.

Why it matters:

- this is the exact file loaded by `run_train_main_pytorch.ipynb`,
- it is the bridge between raw data cleaning and model training,
- it keeps the project reproducible because the training notebook does not depend directly on raw CSV parsing.

## BTC 2023 Dataset Summary

Using the current cleaned bundle:

- raw rows: `3,730,870`
- unique millisecond rows: `3,730,870`
- duplicate millisecond rows: `0`
- cleaned 1-second rows: `935,198`
- average raw rows per second: about `3.99`
- chosen frequency: `1s`

Why this is useful:

- the raw event count is lower than the older sample,
- but the usable 1-second sample count is much larger,
- which makes it much better for the current sequence-based training setup.

## What The Main Training Notebook Does

`run_train_main_pytorch.ipynb` is organized as a complete experiment notebook.

### 1. Load The BTC 2023 Bundle

The notebook:

- loads `data/BTC20230120_clean_1s.pkl`,
- reads the symbol from bundle metadata instead of hardcoding it,
- keeps dataset-specific checkpoint names with the `btc20230120` prefix.

Why this matters:

- the notebook is now tied to the current dataset on purpose,
- it avoids mixing old checkpoints with new runs,
- it avoids symbol mismatches when the source bundle changes.

### 2. Build Features And Sequences

The notebook engineers extra features on top of the 40 FI-2010 fields:

- `mid_price`
- `spread_bps`
- `imbalance_l1`
- `imbalance_l10`
- `depth_total`
- `log_return_1`
- `log_return_5`

It then:

- constructs time-ordered train, validation, and test splits,
- purges overlap around the forecast horizon,
- standardizes features using the train split only,
- creates rolling sequences of shape `sequence_length x feature_dim`.

Why this matters:

- it keeps the evaluation time-consistent,
- it avoids target leakage,
- it gives every model the same input structure.

### 3. Train Return Forecasting Models

The notebook currently includes three return-regression model families:

- `DeepLOBRegressor`
- `LOBTransformerRegressor`
- `HybridLOBTransformerRegressor`

What each one means:

- `DeepLOBRegressor`: convolutional LOB feature extractor plus LSTM temporal head.
- `LOBTransformerRegressor`: a smaller encoder-only transformer over the time window.
- `HybridLOBTransformerRegressor`: DeepLOB-style convolutional stem plus transformer temporal encoder.

Why this matters:

- DeepLOB gives a strong local-structure baseline,
- the pure transformer tests whether attention alone is enough,
- the hybrid tests the more realistic `local structure + temporal attention` idea.

### 4. Train Direction Classification Models

The notebook also includes two direction setups:

- naive direction prediction,
- improved direction prediction.

The improved version changes the labeling logic by:

- using a smoothed future average mid-price,
- adding a dead-zone around zero,
- searching a threshold on validation,
- using weighted cross-entropy.

Why this matters:

- raw direction labels are noisy,
- tiny microstructure moves should often stay `flat`,
- the improved direction task is more aligned with what a trading decision would care about.

## Target And Objective Options

The notebook now exposes more explicit optimization choices.

### Return Targets

- `target_return`: future single-point mid-price return after the prediction horizon.
- `smoothed_return`: future average mid-price return over the prediction horizon.

Why this matters:

- `target_return` is a direct target,
- `smoothed_return` is often less noisy and easier for transformers to learn.

### Return Objectives

- `smooth_l1`: the basic regression objective.
- `weighted_directional`: SmoothL1 plus a directional penalty and extra weight on larger-magnitude targets.

Why this matters:

- plain regression can underweight sign mistakes,
- large moves are more important than tiny moves,
- directional alignment is often useful even in a regression setup.

### Training Stabilizers

The notebook also supports:

- gradient clipping,
- `ReduceLROnPlateau`,
- small validation grid searches for transformer and hybrid settings.

Why this matters:

- transformers can be sensitive to optimizer settings,
- clipping helps prevent unstable updates,
- a plateau scheduler helps when validation stops improving,
- small searches reduce the chance of writing off a model because of one unlucky configuration.

## Current Checkpoint Naming

The BTC 2023 runs now save to dataset-specific checkpoint names such as:

- `best_val_model_pytorch_btc20230120_return_10s.pt`
- `best_val_model_pytorch_btc20230120_transformer_return_10s.pt`
- `best_val_model_pytorch_btc20230120_hybrid_transformer_return_10s.pt`
- `best_val_model_pytorch_btc20230120_direction_naive_10s.pt`
- `best_val_model_pytorch_btc20230120_direction_improved_10s.pt`

Why this matters:

- it prevents silent reuse of checkpoints from older datasets,
- it makes the experiment lineage clearer.

## How To Run The Current BTC 2023 Pipeline

### 1. Clean The BTC 2023 Raw File

Run:

```powershell
python clean_tardis_lob_dataset.py
```

This uses the BTC 2023 defaults and writes:

```text
data/BTC20230120_clean_1s.pkl
```

### 2. Inspect The Data

Open:

```text
sample_adjustment.ipynb
```

Use it to:

- confirm the selected BTC 2023 file,
- inspect timestamp density,
- preview cleaned snapshots before training.

### 3. Train Models

Open:

```text
run_train_main_pytorch.ipynb
```

Run the notebook from top to bottom, or section by section:

- data preparation,
- DeepLOB return forecasting,
- transformer return forecasting,
- Conv-Transformer return forecasting,
- naive direction prediction,
- improved direction prediction.

## Recommended Next Experiment Order

If you want to continue improving the BTC 2023 workflow, the most reasonable order is:

1. keep the BTC 2023 cleaned bundle fixed,
2. try `smoothed_return` as the return target,
3. compare `smooth_l1` against `weighted_directional`,
4. run the hybrid Conv-Transformer search,
5. compare all return models on the same test split,
6. then revisit direction labeling thresholds.

## Scope Note

This README intentionally focuses on the **current BTC 2023 workflow only**. Older files and legacy experiments still exist in the repository, but they are not the recommended starting point for the present project setup.
