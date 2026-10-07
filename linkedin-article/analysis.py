"""Иллюстративный кейс: прогноз выручки оптового дистрибьютора (~$20 млн/год).
ВСЕ ДАННЫЕ СИНТЕТИЧЕСКИЕ - компания вымышленная."""
import json
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
raw.to_csv("raw_export.csv", index=False)

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
df.to_csv("clean_data.csv", index=False)
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
print(summary); summary.to_csv("backtest_summary.csv"); bt.to_csv("backtest_detail.csv", index=False)

# ---------- 5. Прогноз 2026 ----------
final = "ridge"   # разница с ансамблем в пределах шума; берём простую интерпретируемую модель
fidx = pd.DatetimeIndex(fut.month)
tr, te = np.arange(N), np.arange(N, N+12)
pred = fit_predict(final, tr, te)
e = np.log(1 + bt[bt.model == final].err)                              # log(прогноз/факт)
q10, q90 = np.quantile(e, [.1, .9])
lo, hi = pred*np.exp(-q90), pred*np.exp(-q10)
fc = pd.DataFrame({"month": fidx, "forecast": pred.round(0), "lo80": lo.round(0), "hi80": hi.round(0)})
fc.to_csv("forecast_2026.csv", index=False)

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
json.dump(res, open("results.json","w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False)); print(fc)

# ---------- 7. Графики ----------
BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#a8a7a0", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,
    "axes.edgecolor":GRID,"axes.labelcolor":MUTED,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.grid":True,"grid.color":GRID,"grid.linewidth":.8,"axes.axisbelow":True,"figure.facecolor":"white"})
def save(fig, name):
    fig.savefig(f"charts/{name}.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.close(fig)
M = lambda v, p: f"\\${v/1e6:.1f}M"

# Fig1 прогноз
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(df.month, df.revenue/1e6, color=BLUE, lw=2, label="Факт (2022–2025)")
ax.fill_between(fidx, lo/1e6, hi/1e6, color=ORANGE, alpha=.18, lw=0, label="Интервал 80%")
ax.plot(fidx, pred/1e6, color=ORANGE, lw=2, ls=(0,(5,2)), label="Прогноз 2026")
ax.plot([df.month.iloc[-1], fidx[0]], [df.revenue.iloc[-1]/1e6, pred[0]/1e6], color=ORANGE, lw=2, ls=(0,(5,2)))
ax.annotate(f"Пик {res['peak']}: ≈{res['peak_val']/1e6:.2f} млн \\$", (fc.month[fc.forecast.idxmax()], res["peak_val"]/1e6),
            xytext=(-150, 8), textcoords="offset points", color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=GRAY))
ax.set_ylabel("Выручка в месяц, млн \\$"); ax.set_ylim(0.8, None)
ax.set_title("Выручка по месяцам и прогноз на 2026 год", loc="left", fontsize=14, color=INK, pad=18, weight="bold")
ax.axvline(pd.Timestamp('2025-12-15'), color=GRAY, lw=1)
ax.text(0, 1.02, f"2025: {res['rev2025']/1e6:.1f} млн \\$  →  2026 (прогноз): {res['rev2026']/1e6:.1f} млн \\$ ({res['growth']*100:+.0f}%)",
        transform=ax.transAxes, color=MUTED, fontsize=10.5)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, .93)); save(fig, "fig1_forecast")

# Fig2 точность
fig, ax = plt.subplots(figsize=(10, 4.6))
lab = {"naive":"Прежний метод:\nпрошлый год × рост","ridge":"Ridge-регрессия\n(акции, маркетинг, раб. дни)","gbr":"Gradient Boosting\n(на остатках тренда)","blend":"Ансамбль\nRidge + Boosting"}
order = ["naive","ridge","gbr","blend"]; vals = [summary.MAPE[k] for k in order]
cols = [GRAY if k=="naive" else (BLUE if k==final else "#9dbfe9") for k in order]
b = ax.bar([lab[k] for k in order], vals, color=cols, width=.55)
for r, v in zip(b, vals): ax.text(r.get_x()+r.get_width()/2, v+.15, f"{v:.1f}%", ha="center", color=INK, fontsize=12, weight="bold")
ax.set_ylabel("Средняя ошибка (MAPE), %"); ax.grid(axis="x", visible=False); ax.set_ylim(0, max(vals)*1.2)
ax.set_title("Точность прогноза на отложенных данных (бэктест, горизонт 6 мес.)", loc="left", fontsize=14, color=INK, weight="bold", pad=18)
ax.text(0, 1.02, "Чем ниже, тем лучше. Модель обучалась только на данных «до» прогнозируемого периода", transform=ax.transAxes, color=MUTED, fontsize=10.5)
save(fig, "fig2_accuracy")

# Fig5 бэктест: факт против прогноза (последние 2 origin = 2025 год)
fig, ax = plt.subplots(figsize=(10, 4.8))
sel = bt[(bt.origin >= 36)]
last = sel[sel.model == final].groupby("i").last()
nv = sel[sel.model == "naive"].groupby("i").last()
ax.plot(df.month[24:N], df.revenue[24:N]/1e6, color=INK, lw=2, label="Факт")
ax.plot(df.month[last.index], last.pred/1e6, color=BLUE, lw=2, marker="o", ms=4, label=f"Модель (MAPE {summary.MAPE[final]:.1f}% по всему бэктесту)")
ax.plot(df.month[nv.index], nv.pred/1e6, color=GRAY, lw=2, ls=(0,(5,2)), marker="o", ms=4, label=f"Прежний метод (MAPE {summary.MAPE['naive']:.1f}%)")
ax.set_ylabel("Выручка в месяц, млн \\$"); ax.legend(frameon=False, loc="upper left")
ax.set_title("Бэктест: прогноз против факта на данных, которых модель не видела", loc="left", fontsize=14, color=INK, weight="bold", pad=18)
ax.text(0, 1.02, "Показаны прогнозы, сделанные из точек начала 2025 года на 6 месяцев вперёд; маркетинг в тесте — плановый, не фактический", transform=ax.transAxes, color=MUTED, fontsize=10)
save(fig, "fig5_backtest")

# Fig3 сезонность
lrv = np.log(df.revenue.values); tt_ = np.arange(len(df))
detr = lrv - np.polyval(np.polyfit(tt_, lrv, 1), tt_)
prof = np.exp(pd.Series(detr).groupby(df.month.dt.month.values).mean().values); prof = prof/prof.mean()
fig, ax = plt.subplots(figsize=(10, 4.6))
mn = ["Янв","Фев","Мар","Апр","Май","Июн","Июл","Авг","Сен","Окт","Ноя","Дек"]
pc = (prof-1)*100
b = ax.bar(mn, pc, color=[ORANGE if v>=20 else BLUE for v in pc], width=.6)
for r, v in zip(b, pc): ax.text(r.get_x()+r.get_width()/2, v+(1.2 if v>=0 else -1.2), f"{v:+.0f}%".replace("-0%","0%").replace("+0%","0%"), ha="center", va="bottom" if v>=0 else "top", fontsize=9.5, color=INK)
ax.axhline(0, color=GRAY, lw=1); ax.grid(axis="x", visible=False); ax.set_ylabel("Отклонение от среднего месяца, %")
ax.set_ylim(min(pc)-6, max(pc)+7); ax.tick_params(axis="x", pad=18)
ax.set_title("Сезонный профиль: какие месяцы «делают» год", loc="left", fontsize=14, color=INK, weight="bold", pad=18)
ax.text(0, 1.02, f"Оценка модели: акция ≈ {(np.exp(res['coef_promo'])-1)*100:+.0f}% к выручке месяца; +10% маркетинга ≈ {(1.1**res['coef_mkt']-1)*100:+.0f}% (корреляция)", transform=ax.transAxes, color=MUTED, fontsize=10.5)
save(fig, "fig3_seasonality")

# Fig4 эффект
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
b = a1.bar(["Прежний метод", "Модель"], [res["ss_old"]/1e3, res["ss_new"]/1e3], color=[GRAY, BLUE], width=.5)
for r, v in zip(b, [res["ss_old"], res["ss_new"]]): a1.text(r.get_x()+r.get_width()/2, v/1e3+4, f"\\${v/1e3:.0f} тыс.", ha="center", fontsize=12, weight="bold", color=INK)
a1.set_title("Страховой запас, тыс. \\$", loc="left", fontsize=12, color=INK, weight="bold"); a1.grid(axis="x", visible=False); a1.set_ylim(0, res["ss_old"]/1e3*1.2)
a1.text(.5, -.18, f"высвобождается ≈ \\${res['freed']/1e3:.0f} тыс. оборотных средств", transform=a1.transAxes, ha="center", va="top", color=ORANGE, fontsize=10.5, weight="bold")
vals = [uplift*0.28/1e3, mkt_extra/1e3]
b = a2.bar(["Доп. валовая прибыль\nот +20% маркетинга", "Доп. затраты\nна маркетинг"], vals, color=[BLUE, ORANGE], width=.5)
for r, v in zip(b, vals): a2.text(r.get_x()+r.get_width()/2, v+2, f"\\${v:.0f} тыс.", ha="center", fontsize=12, weight="bold", color=INK)
a2.set_title("«Что если» +20% маркетинга (сен–ноя), тыс. \\$", loc="left", fontsize=12, color=INK, weight="bold"); a2.grid(axis="x", visible=False); a2.set_ylim(0, max(vals)*1.2)
a2.text(.5, -.18, "не окупается → бюджет лучше не увеличивать", transform=a2.transAxes, ha="center", va="top", color=INK, fontsize=10.5, weight="bold")
fig.suptitle("Что это даёт бизнесу (модельная оценка)", x=.01, ha="left", fontsize=14, weight="bold", color=INK, y=1.05)
fig.subplots_adjust(wspace=.3); save(fig, "fig4_business_effect")
