"""Illustrative case: monthly sales forecast of a mid-size manufacturer (~$20M/yr).
ALL DATA IS SYNTHETIC - the company is fictional. Run from the repository root:
    python3 src/forecast.py
"""
import json, os
for _d in ("data", "results", "charts"): os.makedirs(_d, exist_ok=True)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)

# ---------- 1. "Выгрузка из ERP" (синтетика) ----------
idx = pd.date_range("2022-01-01", "2026-12-01", freq="MS")           # включая 2026 для плана драйверов
t = np.arange(len(idx))
season = np.array([.82,.85,.95,.98,1.0,.97,.90,.92,1.05,1.12,1.25,1.35]); season /= season.mean()
promo_months = {2022:[4,11], 2023:[3,9,11], 2024:[5,9,11], 2025:[4,9,11], 2026:[4,9,11]}   # весенняя акция "гуляет" по датам
promo = np.array([int(m in promo_months[y]) for y, m in zip(idx.year, idx.month)])
workdays = np.array([np.busday_count(d.date(), (d + pd.offsets.MonthBegin(1)).date()) for d in idx])
drift = np.where(idx.year == 2022, .006, np.where(idx.year == 2023, .002, np.where(idx.year == 2024, .009, .005)))
level = 1.235e6 * np.exp(np.cumsum(drift))                              # неровный рост
mkt = (55 + 1.0*t) * (1 + 0.25*promo + 0.10*idx.month.isin([10,11,12])) * rng.lognormal(0, .12, len(t))   # $ тыс.
rev = level * season[idx.month-1] * (1 + 0.10*promo) * (mkt/mkt.mean())**0.12 * (workdays/21)**0.8 * rng.lognormal(0, .035, len(t))
full = pd.DataFrame({"month": idx, "revenue": rev.round(0), "marketing_k": mkt.round(1), "promo": promo, "workdays": workdays})
N = 48                                                                  # данные есть до 12.2025
raw = full.iloc[:N].copy()
raw.loc[[7, 19, 33], "marketing_k"] = np.nan                            # пропуски
raw.loc[26, "revenue"] *= 10                                            # ошибка ввода
raw = pd.concat([raw, raw.iloc[[40]]], ignore_index=True)               # дубль строки
raw.to_csv("data/raw_export.csv", index=False)

# ---------- 2. Очистка ----------
df = raw.drop_duplicates("month").sort_values("month").reset_index(drop=True)
n_dupes = len(raw) - len(df)
# выбросы: остаток лог-выручки после вычета линейного тренда и медианы по календарному месяцу, порог 6 MAD
lr = np.log(df.revenue.values); tt_ = np.arange(len(df))
r0 = lr - np.polyval(np.polyfit(tt_, lr, 1), tt_)
r1 = r0 - pd.Series(r0).groupby(df.month.dt.month.values).transform("median").values
mad = np.median(np.abs(r1 - np.median(r1)))
out = pd.Series(np.abs(r1 - np.median(r1)) > 6*1.4826*mad)
n_out = int(out.sum())
df.loc[out, "revenue"] = np.nan
df["revenue"] = df.revenue.interpolate()
n_na = int(df.marketing_k.isna().sum())
df["marketing_k"] = df.marketing_k.interpolate()
df.to_csv("data/clean_data.csv", index=False)
print("дубли", n_dupes, "выбросы", n_out, "пропуски", n_na)

# ---------- 3. Признаки: модель предсказывает годовое изменение (YoY) через изменение драйверов ----------
fut = full.iloc[N:].reset_index(drop=True).copy()
fut["marketing_k"] = fut.marketing_k.values                            # плановый маркетинговый бюджет 2026
fut["revenue"] = np.nan
allr = pd.concat([df, fut], ignore_index=True)
LR = np.log(allr.revenue.values)
def yoy_feats(mkt_override=None):
    mk = np.log(allr.marketing_k.values) if mkt_override is None else mkt_override
    X = pd.DataFrame({"d_promo": allr.promo.diff(12), "d_mkt": pd.Series(mk).diff(12),
                      "d_wd": np.log(allr.workdays).diff(12)})
    return X
X_yoy = yoy_feats(); y_yoy = pd.Series(LR).diff(12).values            # цель: log(rev_t / rev_{t-12})
hist = allr.revenue.values

def lin(): return Ridge(alpha=0.05)
def gbr(): return GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=.05, subsample=.8, random_state=0)

def fit_predict(kind, tr, te, X=None, Xte=None):
    X = X_yoy if X is None else X
    Xte = X if Xte is None else Xte
    trn = tr[tr >= 12]
    base = np.array([hist[i-12] for i in te])                          # выручка того же месяца прошлого года (известна всегда)
    if kind == "naive":     # как считали раньше: прошлый год * (1 + рост за последние 12 мес.)
        g = hist[tr[-12:]].sum() / hist[tr[-24:-12]].sum()
        return base * g
    if kind == "ridge":
        return base * np.exp(lin().fit(X.iloc[trn], y_yoy[trn]).predict(Xte.iloc[te]))
    if kind == "gbr":
        g = hist[tr[-12:]].sum() / hist[tr[-24:-12]].sum()
        m = gbr().fit(X.iloc[trn], y_yoy[trn] - np.log(g))
        return base * g * np.exp(m.predict(Xte.iloc[te]))
    if kind == "blend":
        return .5*fit_predict("ridge", tr, te, X, Xte) + .5*fit_predict("gbr", tr, te, X, Xte)

# ---------- 4. Бэктест: скользящее начало, горизонт 6 мес. ----------
H, kinds = 6, ["naive", "ridge", "gbr", "blend"]
X_bt = X_yoy.copy(); X_bt.loc[24:N, "d_mkt"] = np.log(1.05)   # в тесте маркетинг = ПЛАН (прошлый год +5%), а не факт
rows = []
for o in range(24, N-H+1, 3):
    tr, te = np.arange(o), np.arange(o, o+H)
    for k in kinds:
        for i, pi in zip(te, fit_predict(k, tr, te, None, X_bt)):
            rows.append((k, o, i, hist[i], pi))
bt = pd.DataFrame(rows, columns=["model","origin","i","actual","pred"])
bt["ape"] = (bt.pred/bt.actual - 1).abs()*100
bt["err"] = bt.pred/bt.actual - 1
summary = bt.assign(sq=bt.err**2).groupby("model").agg(MAPE=("ape","mean"), sq=("sq","mean")).round(5)
summary["err_std"] = np.sqrt(summary.pop("sq")).round(4)   # RMSE относительной ошибки
BIAS = bt.groupby("model").err.mean()
summary["under10_share"] = bt.assign(u=bt.err < -0.10).groupby("model").u.mean().round(3)
print(summary); summary.to_csv("results/backtest_summary.csv"); bt.to_csv("results/backtest_detail.csv", index=False)

# ---------- 5. Прогноз 2026 ----------
final = "ridge"   # разница с ансамблем в пределах шума; берём простую интерпретируемую модель
fidx = pd.DatetimeIndex(fut.month)
tr, te = np.arange(N), np.arange(N, N+12)
pred = fit_predict(final, tr, te)
e = np.log(1 + bt[bt.model == final].err)                              # log(прогноз/факт)
q10, q90 = np.quantile(e, [.1, .9])
lo, hi = pred*np.exp(-q90), pred*np.exp(-q10)
fc = pd.DataFrame({"month": fidx, "forecast": pred.round(0), "lo80": lo.round(0), "hi80": hi.round(0)})
fc.to_csv("results/forecast_2026.csv", index=False)

# what-if: +20% маркетинга в сен-ноя 2026
mm = (fut.month.dt.month.isin([9,10,11])).values
mk2 = np.log(allr.marketing_k.values).copy(); mk2[N:][mm] += np.log(1.2)
pw = fit_predict(final, tr, te, yoy_feats(mk2))
uplift = (pw - pred).sum(); mkt_extra = (fut.marketing_k.values[mm]*0.2).sum()*1e3
mm_ = mm
# коэффициенты линейной части (для текста)
lm = lin().fit(X_yoy.iloc[np.arange(12, N)], y_yoy[12:N])
coef = dict(zip(X_yoy.columns, lm.coef_))

# ---------- 6. Эффект для бизнеса (модельная оценка) ----------
cogs_m = df.revenue[-12:].mean() * 0.72
z = 1.65                                                              # сервис-уровень ~95%
sd_old, sd_new = summary.err_std["naive"], summary.err_std[final]
ss_old, ss_new = z*sd_old*cogs_m, z*sd_new*cogs_m
res = dict(final=final, rev2025=float(df.revenue[-12:].sum()), rev2026=float(pred.sum()), sc_rev=float(pred[mm].sum()),
           growth=float(pred.sum()/df.revenue[-12:].sum()-1),
           mape_old=float(summary.MAPE["naive"]), mape_new=float(summary.MAPE[final]),
           sd_old=float(sd_old), sd_new=float(sd_new), ss_old=float(ss_old), ss_new=float(ss_new),
           freed=float(ss_old-ss_new), under_old=float(summary.under10_share["naive"]),
           under_new=float(summary.under10_share[final]), n_under_old=int(round(summary.under10_share['naive']*42)), n_under_new=int(round(summary.under10_share[final]*42)), bias_new=float(BIAS[final]), q10=float(q10), q90=float(q90), uplift=float(uplift), uplift_gp=float(uplift*0.28), mkt_extra=float(mkt_extra),
           n_dupes=n_dupes, n_out=n_out, n_na=n_na, peak=fc.loc[fc.forecast.idxmax(),"month"].strftime("%m.%Y"),
           peak_val=float(fc.forecast.max()), coef_mkt=float(coef["d_mkt"]), coef_promo=float(coef["d_promo"]),
           coef_wd=float(coef["d_wd"]), cogs_m=float(cogs_m), n_months=int(len(df)),
           mape_by_model={k: float(v) for k, v in summary.MAPE.items()})
json.dump(res, open("results/results.json","w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False)); print(fc)

# ---------- 7. Charts (EN / UK) ----------
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#a8a7a0", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":GRID,"axes.labelcolor":MUTED,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":.8,"axes.axisbelow":True,"figure.facecolor":"white"})
D = "\\$"
T = {
 "en": dict(
  months=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
  f1_t="Monthly sales and 2026 forecast", f1_s=lambda r: f"2025: {D}{r['rev2025']/1e6:.1f}M  →  2026 (forecast): {D}{r['rev2026']/1e6:.1f}M ({r['growth']*100:+.0f}%)",
  actual="Actual (2022–2025)", band="80% interval", fc="2026 forecast", ylab=f"Sales per month, {D}M",
  peak=lambda r: f"Peak {r['peak']}: ≈{D}{r['peak_val']/1e6:.2f}M",
  f2_t="Forecast accuracy on unseen data (backtest, 6-month horizon)", f2_s="Lower is better. The model is trained only on data from before the forecast period",
  f2_y="Mean error (MAPE), %", lab={"naive":"Old method:\nlast year × growth","ridge":"Ridge regression\n(promos, marketing, work days)","gbr":"Gradient Boosting\n(on trend residuals)","blend":"Ensemble\nRidge + Boosting"},
  f3_t="Seasonal profile: which months make the year", f3_y="Deviation from average month, %",
  f3_s=lambda r: f"Model estimate: a promo campaign ≈ {(np.exp(r['coef_promo'])-1)*100:+.0f}% to monthly sales; +10% marketing ≈ {(1.1**r['coef_mkt']-1)*100:+.0f}% (correlation)",
  f4_t="What it gives the business (modelled estimate)", f4_l=f"Finished-goods safety stock, {D}K", f4_lab=["Old method","Model"], f4_u=lambda v: f"{D}{v/1e3:.0f}K",
  f4_free=lambda r: f"frees up ≈ {D}{r['freed']/1e3:.0f}K of working capital",
  f4_r=f"What-if: +20% marketing (Sep–Nov), {D}K", f4_bars=["Extra gross profit\nfrom +20% marketing","Extra marketing\nspend"], f4_note="does not pay back → do not raise the budget",
  f5_t="Backtest: forecast vs actual on data the model has not seen", f5_s="Forecasts made at the start of 2025, 6 months ahead; marketing in the test is the plan, not the actual",
  f5_a="Actual", f5_m=lambda r: f"Model (MAPE {r:.1f}% over the whole backtest)", f5_o=lambda r: f"Old method (MAPE {r:.1f}%)"),
 "uk": dict(
  months=["Січ","Лют","Бер","Кві","Тра","Чер","Лип","Сер","Вер","Жов","Лис","Гру"],
  f1_t="Продажі за місяцями та прогноз на 2026 рік", f1_s=lambda r: f"2025: {r['rev2025']/1e6:.1f} млн {D}  →  2026 (прогноз): {r['rev2026']/1e6:.1f} млн {D} ({r['growth']*100:+.0f}%)",
  actual="Факт (2022–2025)", band="Інтервал 80%", fc="Прогноз 2026", ylab=f"Продажі за місяць, млн {D}",
  peak=lambda r: f"Пік {r['peak']}: ≈{r['peak_val']/1e6:.2f} млн {D}",
  f2_t="Точність прогнозу на невідомих даних (бектест, горизонт 6 міс.)", f2_s="Чим нижче, тим краще. Модель навчалась лише на даних «до» прогнозованого періоду",
  f2_y="Середня похибка (MAPE), %", lab={"naive":"Старий метод:\nминулий рік × зростання","ridge":"Ridge-регресія\n(акції, маркетинг, роб. дні)","gbr":"Gradient Boosting\n(на залишках тренду)","blend":"Ансамбль\nRidge + Boosting"},
  f3_t="Сезонний профіль: які місяці «роблять» рік", f3_y="Відхилення від середнього місяця, %",
  f3_s=lambda r: f"Оцінка моделі: акція ≈ {(np.exp(r['coef_promo'])-1)*100:+.0f}% до продажів місяця; +10% маркетингу ≈ {(1.1**r['coef_mkt']-1)*100:+.0f}% (кореляція)",
  f4_t="Що це дає бізнесу (модельна оцінка)", f4_l=f"Страховий запас готової продукції, тис. {D}", f4_lab=["Старий метод","Модель"], f4_u=lambda v: f"{D}{v/1e3:.0f} тис.",
  f4_free=lambda r: f"вивільняється ≈ {D}{r['freed']/1e3:.0f} тис. обігових коштів",
  f4_r=f"«Що якщо»: +20% маркетингу (вер–лис), тис. {D}", f4_bars=["Додатковий валовий\nприбуток від +20% маркетингу","Додаткові витрати\nна маркетинг"], f4_note="не окупається → бюджет краще не збільшувати",
  f5_t="Бектест: прогноз проти факту на даних, яких модель не бачила", f5_s="Прогнози з початку 2025 року на 6 місяців уперед; маркетинг у тесті — плановий, а не фактичний",
  f5_a="Факт", f5_m=lambda r: f"Модель (MAPE {r:.1f}% за весь бектест)", f5_o=lambda r: f"Старий метод (MAPE {r:.1f}%)"),
}
def save(fig, lang, name):
    import os; os.makedirs(f"charts/{lang}", exist_ok=True)
    fig.savefig(f"charts/{lang}/{name}.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
def title(ax, t, sub):
    ax.set_title(t, loc="left", fontsize=14, color=INK, weight="bold", pad=18)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=10)

def make_figs(lang):
    L = T[lang]
    # Fig1
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(df.month, df.revenue/1e6, color=BLUE, lw=2, label=L["actual"])
    ax.fill_between(fidx, lo/1e6, hi/1e6, color=ORANGE, alpha=.18, lw=0, label=L["band"])
    ax.plot(fidx, pred/1e6, color=ORANGE, lw=2, ls=(0,(5,2)), label=L["fc"])
    ax.plot([df.month.iloc[-1], fidx[0]], [df.revenue.iloc[-1]/1e6, pred[0]/1e6], color=ORANGE, lw=2, ls=(0,(5,2)))
    ax.axvline(pd.Timestamp("2025-12-15"), color=GRAY, lw=1)
    ax.annotate(L["peak"](res), (fc.month[fc.forecast.idxmax()], res["peak_val"]/1e6), xytext=(-150, 8), textcoords="offset points",
                color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=GRAY))
    ax.set_ylabel(L["ylab"]); ax.set_ylim(0.8, None)
    title(ax, L["f1_t"], L["f1_s"](res)); ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, .93)); save(fig, lang, "fig1_forecast")
    # Fig2
    fig, ax = plt.subplots(figsize=(10, 4.6))
    order = ["naive","ridge","gbr","blend"]; vals = [summary.MAPE[k] for k in order]
    cols = [GRAY if k=="naive" else (BLUE if k==final else "#9dbfe9") for k in order]
    b = ax.bar([L["lab"][k] for k in order], vals, color=cols, width=.55)
    for r_, v in zip(b, vals): ax.text(r_.get_x()+r_.get_width()/2, v+.15, f"{v:.1f}%", ha="center", color=INK, fontsize=12, weight="bold")
    ax.set_ylabel(L["f2_y"]); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(vals)*1.2)
    title(ax, L["f2_t"], L["f2_s"]); save(fig, lang, "fig2_accuracy")
    # Fig3
    fig, ax = plt.subplots(figsize=(10, 4.6))
    lrv = np.log(df.revenue.values); tt_ = np.arange(len(df))
    detr = lrv - np.polyval(np.polyfit(tt_, lrv, 1), tt_)
    prof = np.exp(pd.Series(detr).groupby(df.month.dt.month.values).mean().values); prof = prof/prof.mean(); pc = (prof-1)*100
    b = ax.bar(L["months"], pc, color=[ORANGE if v>=20 else BLUE for v in pc], width=.6)
    for r_, v in zip(b, pc): ax.text(r_.get_x()+r_.get_width()/2, v+(1.2 if v>=0 else -1.2), f"{v:+.0f}%".replace("-0%","0%").replace("+0%","0%").replace("-","−"), ha="center", va="bottom" if v>=0 else "top", fontsize=9.5, color=INK)
    ax.axhline(0, color=GRAY, lw=1); ax.grid(axis="x", visible=False); ax.set_ylabel(L["f3_y"])
    ax.set_ylim(min(pc)-6, max(pc)+7); ax.tick_params(axis="x", pad=10)
    title(ax, L["f3_t"], L["f3_s"](res)); save(fig, lang, "fig3_seasonality")
    # Fig4
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
    sv = [res["ss_old"], res["ss_new"]]
    b = a1.bar(L["f4_lab"], [v/1e3 for v in sv], color=[GRAY, BLUE], width=.5)
    for r_, v in zip(b, sv): a1.text(r_.get_x()+r_.get_width()/2, v/1e3+4, L["f4_u"](v), ha="center", fontsize=12, weight="bold", color=INK)
    a1.set_title(L["f4_l"], loc="left", fontsize=11, color=INK, weight="bold"); a1.grid(axis="x", visible=False); a1.set_ylim(0, sv[0]/1e3*1.2)
    a1.text(.5, -.18, L["f4_free"](res), transform=a1.transAxes, ha="center", va="top", color=ORANGE, fontsize=10.5, weight="bold")
    vals = [uplift*0.28/1e3, mkt_extra/1e3]
    b = a2.bar(L["f4_bars"], vals, color=[BLUE, ORANGE], width=.5)
    for r_, v in zip(b, vals): a2.text(r_.get_x()+r_.get_width()/2, v+2, f"{D}{v:.0f}K" if lang=="en" else f"{D}{v:.0f} тис.", ha="center", fontsize=12, weight="bold", color=INK)
    a2.set_title(L["f4_r"], loc="left", fontsize=11, color=INK, weight="bold"); a2.grid(axis="x", visible=False); a2.set_ylim(0, max(vals)*1.2)
    a2.text(.5, -.18, L["f4_note"], transform=a2.transAxes, ha="center", va="top", color=INK, fontsize=10.5, weight="bold")
    fig.suptitle(L["f4_t"], x=.01, ha="left", fontsize=14, weight="bold", color=INK, y=1.03); fig.subplots_adjust(wspace=.35); save(fig, lang, "fig4_business_effect")
    # Fig5
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sel = bt[bt.origin >= 36]; lastp = sel[sel.model == final].groupby("i").last(); nv = sel[sel.model == "naive"].groupby("i").last()
    ax.plot(df.month[24:N], df.revenue[24:N]/1e6, color=INK, lw=2, label=L["f5_a"])
    ax.plot(df.month[lastp.index], lastp.pred/1e6, color=BLUE, lw=2, marker="o", ms=4, label=L["f5_m"](summary.MAPE[final]))
    ax.plot(df.month[nv.index], nv.pred/1e6, color=GRAY, lw=2, ls=(0,(5,2)), marker="o", ms=4, label=L["f5_o"](summary.MAPE["naive"]))
    ax.set_ylabel(L["ylab"]); ax.legend(frameon=False, loc="upper left"); title(ax, L["f5_t"], L["f5_s"]); save(fig, lang, "fig5_backtest")

for lg in ("en", "uk"): make_figs(lg)
