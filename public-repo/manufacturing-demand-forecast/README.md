# Manufacturing demand forecasting with scikit-learn

🇬🇧 English · [🇺🇦 Українська](README.uk.md)

A worked example of monthly sales forecasting for a mid-size manufacturer (about **$20M** annual sales): data cleaning, a rolling-origin backtest against the "last year × growth" rule, a 12-month forecast with an uncertainty range, and a marketing what-if scenario.

> **All data in this repository is synthetic and the company is fictional.** The code demonstrates a method and the order of magnitude of the results; it is **not** the outcome of a real client engagement. Results on real data will differ, usually for the worse.

![Forecast](charts/en/fig1_forecast.png)

## Key results (on the synthetic data)

| | Old method ("last year × growth") | Ridge model |
|---|---|---|
| Mean forecast error (MAPE, rolling backtest) | 6.7% | 4.9% |
| Months with demand under-forecast by >10% | 5 of 42 | 2 of 42 |
| Calculated finished-goods safety stock | ≈ $175K | ≈ $115K (one-off ≈ $60K released) |

2026 forecast: ≈ **$21.6M** (+8%). A "+20% marketing in Sep–Nov" scenario adds ≈ $36K of gross profit for $84K of spend, so it does not pay back in this example.

## How it works

1. **Synthetic ERP export:** 48 months of sales, marketing budget, promotion calendar and working days, with deliberately injected defects (1 duplicate, 3 gaps, 1 ×10 outlier).
2. **Cleaning:** duplicate removal, interpolation, outlier detection on residuals after removing trend and seasonality (6 MAD).
3. **Target = year-over-year change.** Models predict `log(sales_t / sales_{t-12})` from changes in drivers (promo flag, log marketing budget, log working days).
4. **Models:** naive baseline, `Ridge`, `GradientBoostingRegressor`, 50/50 ensemble.
5. **Validation:** rolling-origin backtest, 6-month horizon, 7 origins, step 3 months; in the test, marketing is the *plan* (last year +5%), not the actual figure.
6. **Forecast:** 2026, 80% interval from empirical backtest log-errors.
7. **Business translation:** safety stock = `z · RMSE · monthly COGS` (z = 1.65, 1-month lead time, COGS = 72% of sales) and a marketing what-if at 28% gross margin.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 src/forecast.py          # run from the repository root
```

The script is deterministic (fixed seed) and rewrites `data/`, `results/` and `charts/{en,uk}/`.

## Repository layout

```
.
├── README.md / README.uk.md
├── LICENSE
├── requirements.txt
├── src/forecast.py              # whole pipeline: data, cleaning, models, backtest, charts
├── data/
│   ├── raw_export.csv           # synthetic export with injected defects
│   └── clean_data.csv           # after cleaning
├── results/
│   ├── backtest_summary.csv     # metrics per model
│   ├── backtest_detail.csv      # every backtest forecast
│   ├── forecast_2026.csv        # forecast with 80% interval
│   └── results.json             # key numbers used in the articles
├── charts/{en,uk}/              # 5 charts per language
└── articles/                    # full write-ups (EN, UK)
```

## Limitations

- Synthetic data; only 48 monthly observations.
- Outlier threshold and model choice were tuned on the same data; no separate hold-out.
- The 80% interval is built from 42 overlapping backtest errors and is most likely too narrow.
- The marketing effect is a correlation in the (synthetic) history, not proven causation, and is calculated without time lags.
- The safety-stock estimate uses company-level sales; per-SKU error is higher, so real savings must be calculated per SKU.
- The forecast depends on the marketing and promotion plan being known in advance.

## Articles

- [English article](articles/article.en.md)
- [Українська стаття](articles/article.uk.md)
- LinkedIn: _add the link to your published article here_

## License

[MIT](LICENSE)
