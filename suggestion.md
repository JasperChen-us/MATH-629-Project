# Suggestion

## Data Collection

- For longer LOB collection, prefer Binance WebSocket market-data streams over repeated REST polling. The best next step is `diff. depth` or `partial book depth` streams so you capture updates continuously instead of one snapshot per second.
- The current project now has a partial-depth WebSocket collector, but the stronger production upgrade is full local-book reconstruction from `diff depth` updates plus an initial snapshot.
- Keep everything in UTC and store the exchange event time as the primary clock.
- Run the collector for at least several hours, and ideally several full trading days, before trusting model metrics or strategy metrics.
- Save rolling checkpoints to disk every few minutes. For long runs, do not wait until the very end to write one final pickle.
- Keep two datasets:
  - raw event-level LOB updates
  - resampled model-ready snapshots such as 1-second or 5-second bars
- Start with `BTCUSDT`, `ETHUSDT`, and `DOGEUSDT`, then add more liquid pairs only after the pipeline is stable.
- For historical bootstrap data, consider:
  - Tardis.dev
  - DataBento
  - CoinAPI
  - Kaggle research datasets
- A good workflow is:
  - use a historical provider or sample dataset to prototype faster
  - use Binance WebSocket for your own live collection
  - compare the live collector output against a historical source for sanity checks

## Exchange Access Options

- WebSocket feeds:
  - best for real-time L2/L3 collection
  - lowest latency and most scalable path for long-running collection
- Historical data providers:
  - useful because exchange public endpoints often do not give full historical LOB replay directly
  - best for backfilling research windows quickly
- Data aggregators:
  - helpful if you want one schema across multiple exchanges
- Kaggle datasets:
  - good for early experiments and architecture checks
  - less ideal for production because preprocessing choices are fixed by someone else

## Reconstruction

- For higher-quality LOB research, reconstruct the local order book from diff-depth streams instead of relying only on top-level snapshots.
- This usually means:
  - request an initial book snapshot
  - subscribe to exchange diff-depth updates
  - apply updates in order
  - periodically checkpoint the reconstructed book to disk
- That reconstruction path should be the next collector upgrade if you want deeper microstructure features such as queue evolution, cancellations, and more stable imbalance signals.

## Modeling

- The current notebook now predicts future mid-price, but for training stability you may eventually want to predict:
  - future log mid-price
  - future mid-price change
  - future return
- A strong compromise is to optimize on future return and then convert back to mid-price for plotting and trading.
- Try multi-horizon outputs such as 1s, 5s, and 10s ahead in one model. This usually gives a richer signal than a single horizon.
- Add normalization that is fit only on the training split and applied symbol-by-symbol, exactly as the current regression pipeline does.
- Good extra features to try next:
  - queue imbalance per level
  - top-of-book price changes
  - rolling volatility
  - rolling spread mean
  - depth slope / convexity
  - realized order flow imbalance

## Validation

- Use walk-forward validation instead of one fixed split once the dataset gets larger.
- Track both regression metrics and trading metrics. Low RMSE alone does not guarantee useful signals.
- Add a naive benchmark such as:
  - next mid-price equals current mid-price
  - moving-average forecast
  - linear regression on engineered features

## Strategy

- Add transaction costs and slippage before trusting any PnL result.
- Use a signal threshold so the strategy only trades when the predicted move is larger than noise.
- Compare:
  - always-in long/short
  - thresholded long/short
  - top-quantile signals only
- Report gross and net metrics separately.

## Practical Next Step

- The biggest upgrade is a long-running WebSocket collector with rolling checkpoint writes. Once you have that, the current notebook should become much more meaningful.
