# LinkedIn teaser posts

Attach `charts/<lang>/fig1_forecast.png`. Replace `[LINK]` with the link to your article (and, if you like, the GitHub repository).

## English

Most manufacturers plan production "by last year plus growth".

I tested that rule against a simple scikit-learn model on a modelled $20M manufacturer (synthetic data, for demonstration):

📉 forecast error: 6.7% → 4.9% on a rolling backtest
📦 ≈ $60K of working capital released from finished-goods safety stock (one-off)
💡 a "+20% marketing" scenario that, per the model, does not pay back, checked before the money was spent

What made the difference: predicting the year-over-year change driven by the promo calendar, marketing and working days, instead of the level itself. Honest validation: the model only ever saw data from before the period it forecast.

The full write-up (data cleaning, methods, limitations, charts, code): [LINK]

How do you plan production today? 👇

#DataScience #Forecasting #scikitlearn #Manufacturing #SupplyChain

---

## Українська

Більшість виробників планує випуск «за минулим роком плюс зростання».

Я порівняв це правило з простою моделлю на scikit-learn на модельному підприємстві з оборотом $20 млн (синтетичні дані, для демонстрації):

📉 похибка прогнозу: 6,7% → 4,9% на ковзному бектесті
📦 ≈ $60 тис. обігових коштів вивільняється зі страхового запасу готової продукції (разово)
💡 сценарій «+20% маркетингу» за моделлю не окупається — це видно ще до того, як гроші витрачено

Що дало результат: прогноз не рівня продажів, а річної зміни за календарем акцій, маркетингом і робочими днями. Чесна валідація: модель бачила лише дані «до» прогнозованого періоду.

Повний розбір (очищення даних, методи, обмеження, графіки, код): [LINK]

Як ви плануєте виробництво сьогодні? 👇

#DataScience #Forecasting #scikitlearn #Manufacturing #SupplyChain
