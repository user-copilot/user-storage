# Sales forecasting case study (synthetic data)

Illustrative case for a LinkedIn article: monthly sales forecast of a mid-size manufacturer (~$20M/yr) with scikit-learn.
**All data is synthetic**; the company is fictional. The code shows the method, not a real client result.

## Run

```bash
pip install -r requirements.txt
python3 analysis.py
```

The script generates the synthetic "ERP export", cleans it, runs a rolling backtest (naive baseline, Ridge, GradientBoosting, ensemble) on year-over-year changes, forecasts 2026, runs a marketing what-if, and writes charts to `charts/en` and `charts/uk`.

## Files

| File | Content |
|---|---|
| `analysis.py` | whole pipeline (data generation, cleaning, models, backtest, charts) |
| `raw_export.csv`, `clean_data.csv` | synthetic data before / after cleaning |
| `backtest_summary.csv`, `backtest_detail.csv` | backtest metrics and every forecast |
| `forecast_2026.csv`, `results.json` | forecast with 80% interval and key numbers |
| `article.en.md`, `article.uk.md` | article texts |
| `teaser_posts.md` | short LinkedIn posts (EN / UK) |

## Known limitations

Synthetic data; 48 monthly observations; outlier threshold and model choice tuned on the same data; interval from only 42 overlapping backtest errors; marketing effect is a correlation; safety-stock saving is a rough estimate on company-level sales.
