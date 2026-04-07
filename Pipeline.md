# Pipeline

## Why Raw Future Mid-Price Level Is A Weak Primary Target

In high-frequency LOB data, the mid-price level is highly persistent. If the model predicts

`future_mid_price ~= current_mid_price`

it can already look good on plots and on price-level RMSE, especially after second-level resampling. That creates two problems:

- the task can look easier than it really is
- good price-level error does not necessarily mean useful trading signal

This is why the pipeline should not treat raw future mid-price level as the main optimization target.

## Better Target Definitions

### Recommended default

Use **future return over a fixed horizon** as the main regression target:

`target_return = (mid_price_t+h - mid_price_t) / mid_price_t`

Why:

- scale-free across time and instruments
- directly tied to trading decisions
- easier to compare against zero-return / persistence baselines

### Also useful

- **Future mid-price change**
  - `mid_price_t+h - mid_price_t`
  - good when you want price reconstruction and dollar/tick interpretation
- **Price direction classification**
  - sign of the future return
  - useful when direction matters more than magnitude
  - should usually include a neutral class for small moves
- **Event or movement labels**
  - label `{-1, 0, +1}` using a return threshold
  - small moves are treated as noise
  - this is often better aligned with execution decisions than raw sign labels

## Dataset Changes

- Keep the cleaned UTC LOB snapshots.
- Keep the existing 40 FI-2010-style LOB inputs plus a small set of engineered features.
- Build labels from a forecast horizon such as `10s`, `20s`, or `30s`, not only `1s`.
- Add a train-set movement threshold from the absolute future return distribution.
  - example: 75th percentile of `abs(target_return)` on the training split
- Store all of these in the sequence metadata:
  - `current_mid_price`
  - `target_mid_price`
  - `target_mid_price_delta`
  - `target_return`
  - `target_log_return`
  - `actual_direction`
  - `actual_movement_label`
  - `movement_threshold`

## Label Changes

Use return as the default target for training, and derive the rest from it:

- training target:
  - `target_return`
- evaluation labels:
  - `actual_direction = sign(target_return)`
  - `actual_movement_label = movement_label(target_return, threshold)`

This keeps one main target while still supporting direction and event-style evaluation.

## Loss Changes

Use a robust regression loss on the chosen target instead of plain MSE on price level.

Recommended:

- `SmoothL1Loss` / Huber-style loss for return regression

Why:

- less sensitive to large outliers
- better fit for noisy microstructure targets

## Evaluation Changes

Do not rely only on price-level MSE or RMSE.

Track:

- **Price reconstruction**
  - price MAE
  - price RMSE
  - price MAPE
- **Target quality**
  - return MAE in bps
  - return RMSE in bps
  - Pearson information coefficient between predicted and realized return
  - Spearman rank IC
- **Decision quality**
  - direction accuracy
  - movement accuracy
  - active precision / recall / F1
  - predicted active share
  - actual active share
- **Trading usefulness**
  - thresholded backtest
  - turnover
  - annualized return
  - Sharpe
  - PnL path

## Strong Baselines

Always compare against simple baselines before trusting a deep model.

Minimum set:

- **Persistence / zero-return**
  - predict no change
  - equivalent to last mid-price for return forecasting
- **Train-mean target**
  - predict the unconditional average future return
- **Linear model on engineered features**
  - recommended next baseline
- **Rolling mean return baseline**
  - recommended next baseline

For high-frequency data, beating persistence slightly can already be meaningful. Failing to beat it is a strong warning sign.

## Practical Recommendation

For this repo, the most sensible default is:

1. train on **future return**
2. use a **movement threshold** from the training split
3. evaluate against **persistence** and **train-mean** baselines
4. still reconstruct future mid-price for plots and backtests

## Applied In Code

The current codebase has been updated accordingly:

- the pipeline now supports target-aware reconstruction and defaults cleanly to return forecasting when requested
- the notebook now uses:
  - `target_mode='return'`
  - `target_horizon=10`
  - `SmoothL1Loss`
  - baseline benchmarking instead of price-level-only reporting
- movement-aware labels are created from train-set return thresholds

## Next Good Upgrade

If you want one more step after this, the best one is:

- add a simple linear regression baseline on the same features
- compare horizons side by side, such as `10s`, `20s`, and `30s`

That will tell you whether the deep model is truly adding signal or just matching persistence.
