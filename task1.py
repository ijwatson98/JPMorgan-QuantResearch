# -*- coding: utf-8 -*-
"""
JPMorgan Chase & Co. Quantitative Research Virtual Experience Program
Task 1

Investigating and Analysing Price Data
"""

# =============================================================================
# Exploratory Analysis
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime as dt

gas_data = pd.read_csv('Nat_Gas.csv')
gas_data['Dates'] = pd.to_datetime(gas_data['Dates'])

ax = gas_data.plot('Dates', 'Prices', style='x-')

def split_dates(date_df):  
    X = date_df.copy()
    X['Ordinal Dates'] = X['Dates'].apply(lambda x: dt.datetime.toordinal(x))
    X['DayOfWeek'] = X['Dates'].dt.dayofweek
    X['DayOfMonth'] = X['Dates'].dt.day
    X['Month'] = X['Dates'].dt.month
    X['Quarter'] = X['Dates'].dt.quarter
    X['DayOfYear'] = X['Dates'].dt.dayofyear
    X['Year'] = X['Dates'].dt.year
    X = X.set_index('Dates')
    return X

X = split_dates(gas_data).drop('Prices', axis=1)
y = gas_data.set_index('Dates')[['Prices']]

split = round(len(X)*0.66)
X_train, X_test = X.iloc[:split, :], X.iloc[split:, :]
y_train, y_test = y.iloc[:split, :], y.iloc[split:, :]

lr = LinearRegression()
lr.fit(X_train[['Ordinal Dates']], y_train)

y_fit = pd.DataFrame(
    lr.predict(X_train[['Ordinal Dates']]),
    index=y_train.index,
    columns=y_train.columns,
)

y_pred = pd.DataFrame(
    lr.predict(X_test[['Ordinal Dates']]),
    index=y_test.index,
    columns=y_test.columns,
)

axs = y_train.plot(color='0.25', subplots=True, sharex=True)
axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit.plot(color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred.plot(color='C3', subplots=True, sharex=True, ax=axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Trends")


# =============================================================================
# Seasonality Analysis
# =============================================================================

import seaborn as sns

def seasonal_plot(X, y, period, freq, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    palette = sns.color_palette("husl", n_colors=X[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=X,
        ci=False,
        ax=ax,
        palette=palette,
        legend=False,
    )
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    for line, name in zip(ax.lines, X[period].unique()):
        y_ = line.get_ydata()[-1]
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center",
        )
    return ax

seasonal_plot(X, np.array(y).reshape(-1,), period="Year", freq="Month")


# =============================================================================
# Residual Analysis (XGBoost)
# =============================================================================

y_resid = y_train-y_fit
plt.plot(y_resid)

from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(X_train, y_resid)

from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
from hyperopt.pyll import scope
from sklearn.metrics import mean_squared_error

def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials
                            if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['model']

def objective(space):
    
    model = XGBRegressor(
        max_depth=space['max_depth'],
        min_child_weight=space['min_child_weight'],
        n_estimators=space['n_estimators'],
        eta=space['eta'],
        eval_metric="merror"
    )
    
    model.fit(X_train, y_resid)
    
    y_pred_boosted = model.predict(X_test) + np.array(y_pred).reshape(-1,)
        
    loss = mean_squared_error(y_test, y_pred_boosted, squared=False)
            
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

space={
    'max_depth': scope.int(hp.quniform("max_depth", 1, 5, 1)),
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 5, 20, 1)),
    'n_estimators': scope.int(hp.quniform("n_estimators", 80, 100, 5)),
    'eta': hp.quniform("eta", 0.1, 0.4, 0.05)
}

trials = Trials()

best_params = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    trials=trials,
    max_evals=1000)

xgb = getBestModelfromTrials(trials)

y_fit_boosted = pd.DataFrame(
    xgb.predict(X_train) + np.array(y_fit).reshape(-1,),
    index=y_train.index,
    columns=y_train.columns,
)

y_pred_boosted = pd.DataFrame(
    xgb.predict(X_test) + np.array(y_pred).reshape(-1,),
    index=y_test.index,
    columns=y_test.columns,
)

axs = y_train.plot(color='0.25', subplots=True, sharex=True)
axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit_boosted.plot(color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred_boosted.plot(color='C3', subplots=True, sharex=True, ax=axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Hybrid")

all_data = pd.concat([y_fit_boosted, y_pred_boosted])

# =============================================================================
# Hybrid Forecast
# =============================================================================

fc_dates_monthly = pd.DataFrame({'Dates': pd.date_range('10/01/2024', '09/30/2025', freq='M')})
fc_dates_monthly = split_dates(fc_dates_monthly)

y_forecast_boosted = pd.DataFrame({'Dates': np.array(lr.predict(fc_dates_monthly[['Ordinal Dates']]))
                                   .reshape(-1,) + xgb.predict(fc_dates_monthly)},
    index=fc_dates_monthly.index
)

axs = y_train.plot(color='0.25', subplots=True, sharex=True)
axs = y_test.plot(color='0.25', subplots=True, sharex=True, ax=axs)
axs = y_fit_boosted.plot(color='C0', subplots=True, sharex=True, ax=axs)
axs = y_pred_boosted.plot(color='C3', subplots=True, sharex=True, ax=axs)
axs = y_forecast_boosted.plot(color='C2', subplots=True, sharex=True, ax=axs)
for ax in axs: ax.legend([])
_ = plt.suptitle("Monthly Forecast")

all_data = pd.concat([all_data, y_forecast_boosted])

# =============================================================================
# Price Predictor Forecast
# =============================================================================

def price_predictor(date: str):
    data_df = pd.DataFrame({'Dates': pd.to_datetime(date)}, index=[0])
    data_df = split_dates(data_df)
    return float(np.array(lr.predict(data_df[['Ordinal Dates']])).reshape(-1,) + xgb.predict(data_df))

print(price_predictor("20/09/26"))

# =============================================================================
# Daily Data
# =============================================================================

daily_data = pd.DataFrame({'Dates': pd.date_range('10/31/2020', '09/30/2025', freq='D')})
daily_data['Prices'] = daily_data['Dates'].apply(price_predictor)
daily_data = daily_data.set_index('Dates')
daily_data.to_csv("daily_data.csv")


