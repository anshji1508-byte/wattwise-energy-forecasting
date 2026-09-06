# %% [markdown]
# # Appliances Energy Forecasting — Advanced Time-Series ML Workflow
#
# This notebook treats the problem as **one-step-ahead forecasting**. Every feature
# is available before the value being predicted. Metrics are computed on future data.

# %%
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DATA_PATH = Path('energydata_complete.csv')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# %% [markdown]
# ## 1. Business framing and validation design
#
# Target: `Appliances` energy consumption. This is regression, not classification.
# We keep chronological order: first 80% for training, last 20% as unseen future data.
# Random splitting is intentionally avoided because it allows future patterns into train.

# %%
raw = pd.read_csv(DATA_PATH)
raw['date'] = pd.to_datetime(raw['date'], errors='coerce')
raw = raw.dropna(subset=['date']).sort_values('date').drop_duplicates().reset_index(drop=True)
target_name = 'Appliances'
print(f'Rows: {len(raw):,} | Columns: {raw.shape[1]}')
print(f'Date range: {raw.date.min()} -> {raw.date.max()}')
print(f'Missing cells: {raw.isna().sum().sum()} | Duplicate rows: {raw.duplicated().sum()}')
display(raw.head(), raw.describe(include='all').T.head(10))

# %% [markdown]
# ## 2. EDA: trend, seasonality, outliers and correlations
#
# High consumption values are not removed: they may be real appliance events.

# %%
eda = raw.set_index('date')
fig, axes = plt.subplots(2, 2, figsize=(17, 9))
eda[target_name].plot(ax=axes[0,0], linewidth=.45, color='steelblue')
axes[0,0].set_title('Target through time'); axes[0,0].set_ylabel('Consumption')
sns.histplot(raw[target_name], bins=60, kde=True, ax=axes[0,1], color='darkorange')
axes[0,1].set_title('Target distribution')
hour = raw.assign(hour=raw.date.dt.hour).groupby('hour')[target_name].mean()
hour.plot(kind='bar', ax=axes[1,0], color='seagreen'); axes[1,0].set_title('Mean by hour')
dow = raw.assign(day=raw.date.dt.dayofweek).groupby('day')[target_name].mean()
dow.plot(kind='bar', ax=axes[1,1], color='mediumpurple'); axes[1,1].set_title('Mean by day (0=Monday)')
plt.tight_layout(); plt.show()

print('Target quantiles:')
display(raw[target_name].quantile([.01,.25,.5,.75,.95,.99,.999]).to_frame('consumption'))
numeric_corr = raw.select_dtypes('number').corr()[target_name].sort_values(ascending=False)
display(numeric_corr.head(12).to_frame('correlation'))

# %% [markdown]
# ## 3. Leakage-safe feature engineering
#
# The dataset is recorded every 10 minutes: 6 rows/hour, 144 rows/day, 1008 rows/week.
# Lag and rolling features use `shift(1)`, so the current target never leaks into its own features.

# %%
def make_features(frame):
    z = frame.copy().sort_values('date').reset_index(drop=True)
    dt = z['date']
    z['month'] = dt.dt.month
    z['day_of_month'] = dt.dt.day
    z['day_of_week'] = dt.dt.dayofweek
    z['hour'] = dt.dt.hour
    z['minute'] = dt.dt.minute
    z['is_weekend'] = (z['day_of_week'] >= 5).astype(int)
    slot = z['hour'] * 6 + z['minute'] // 10
    z['time_sin'] = np.sin(2*np.pi*slot/144)
    z['time_cos'] = np.cos(2*np.pi*slot/144)
    z['dow_sin'] = np.sin(2*np.pi*z['day_of_week']/7)
    z['dow_cos'] = np.cos(2*np.pi*z['day_of_week']/7)
    y = z[target_name].copy()
    for lag in [1, 2, 3, 6, 12, 18, 36, 72, 144, 288, 1008]:
        z[f'lag_{lag}'] = y.shift(lag)
    for window in [6, 18, 36, 144]:
        past = y.shift(1)
        z[f'rolling_mean_{window}'] = past.rolling(window).mean()
        z[f'rolling_std_{window}'] = past.rolling(window).std()
    y_out = z[target_name]
    X_out = z.drop(columns=['date', target_name, 'rv2'], errors='ignore')
    valid = X_out.notna().all(axis=1)
    return X_out.loc[valid].reset_index(drop=True), y_out.loc[valid].reset_index(drop=True), z.loc[valid, 'date'].reset_index(drop=True)

X, y, dates = make_features(raw)
split = int(len(X) * .8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]
print(f'Feature matrix: {X.shape} | train: {X_train.shape} | future test: {X_test.shape}')

# %% [markdown]
# ## 4. Baselines and chronological cross-validation
#
# The naive forecast (last known value) is mandatory. A complex model is useful only
# if it beats this baseline on future data.

# %%
def make_pipeline(estimator):
    return Pipeline([('imputer', SimpleImputer(strategy='median')),
                     ('scaler', RobustScaler()), ('model', estimator)])

def metrics(actual, predicted):
    return {'R2': r2_score(actual, predicted),
            'MAE': mean_absolute_error(actual, predicted),
            'RMSE': mean_squared_error(actual, predicted) ** .5}

def cv_score(estimator, X_data=X_train, y_data=y_train, n_splits=4):
    rows = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X_data):
        model = make_pipeline(clone(estimator))
        model.fit(X_data.iloc[train_idx], y_data.iloc[train_idx])
        rows.append(metrics(y_data.iloc[val_idx], model.predict(X_data.iloc[val_idx])))
    return pd.DataFrame(rows).mean()

candidate_models = {
    'Ridge': Ridge(alpha=10.0),
    'Extra Trees': ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, max_features=.8, random_state=RANDOM_STATE, n_jobs=-1),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=18, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1),
    'Hist Gradient Boosting': HistGradientBoostingRegressor(max_iter=250, learning_rate=.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=RANDOM_STATE),
}
cv_results = []
for name, estimator in candidate_models.items():
    score = cv_score(estimator)
    cv_results.append([name, score.R2, score.MAE, score.RMSE])
cv_table = pd.DataFrame(cv_results, columns=['Model','R2','MAE','RMSE']).sort_values('RMSE')
display(cv_table.style.format({'R2':'{:.4f}','MAE':'{:.2f}','RMSE':'{:.2f}'}))

# %% [markdown]
# ## 5. Optuna hyperparameter tuning
#
# Optuna minimises chronological validation RMSE. The test set remains untouched until the end.

# %%
def objective(trial):
    params = {
        'max_iter': trial.suggest_int('max_iter', 80, 180),
        'learning_rate': trial.suggest_float('learning_rate', .02, .12, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 15, 63),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 80),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 20, log=True),
    }
    scores = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=2).split(X_train):
        model = make_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params))
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[va_idx])
        scores.append(mean_squared_error(y_train.iloc[va_idx], pred) ** .5)
    return float(np.mean(scores))

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=8, show_progress_bar=False)
print('Best CV RMSE:', round(study.best_value, 3))
print('Best parameters:', study.best_params)
display(optuna.visualization.matplotlib.plot_optimization_history(study)); plt.show()

# %% [markdown]
# ## 6. Final future holdout score

# %%
tuned_model = make_pipeline(HistGradientBoostingRegressor(random_state=RANDOM_STATE, **study.best_params))
tuned_model.fit(X_train, y_train)
pred = tuned_model.predict(X_test)
naive = X_test['lag_1'].to_numpy()
final_scores = pd.DataFrame([
    ['Naive previous value', *metrics(y_test, naive).values()],
    ['Tuned Hist Gradient Boosting', *metrics(y_test, pred).values()],
], columns=['Model','R2','MAE','RMSE'])
display(final_scores.style.format({'R2':'{:.4f}','MAE':'{:.2f}','RMSE':'{:.2f}'}))

# %% [markdown]
# ## 7. Error analysis
#
# We inspect residual distribution, error over time, error by hour, and the worst misses.
# This tells us where the model fails instead of hiding behind one score.

# %%
errors = pd.DataFrame({'date': dates.iloc[split:].values, 'actual': y_test.values,
                       'predicted': pred, 'error': y_test.values - pred})
errors['abs_error'] = errors['error'].abs()
errors['hour'] = errors.date.dt.hour
errors['day_of_week'] = errors.date.dt.dayofweek
errors['pct_error'] = errors['abs_error'] / np.maximum(errors['actual'], 1) * 100
fig, ax = plt.subplots(2, 2, figsize=(17, 9))
ax[0,0].plot(errors.date.iloc[:600], errors.actual.iloc[:600], label='Actual', linewidth=1)
ax[0,0].plot(errors.date.iloc[:600], errors.predicted.iloc[:600], label='Predicted', linewidth=1)
ax[0,0].set_title('Actual vs prediction (first 600 future points)'); ax[0,0].legend()
sns.histplot(errors.error, bins=50, kde=True, ax=ax[0,1]); ax[0,1].axvline(0, color='black'); ax[0,1].set_title('Residual distribution')
sns.boxplot(data=errors, x='hour', y='abs_error', ax=ax[1,0]); ax[1,0].set_title('Absolute error by hour')
ax[1,1].scatter(errors.actual, errors.predicted, s=8, alpha=.25)
lims=[errors.actual.min(), errors.actual.max()]; ax[1,1].plot(lims, lims, 'r--'); ax[1,1].set_xlabel('Actual'); ax[1,1].set_ylabel('Predicted'); ax[1,1].set_title('Calibration plot')
plt.tight_layout(); plt.show()
print('Residual mean:', round(errors.error.mean(), 3), '| Positive means under-prediction.')
display(errors.sort_values('abs_error', ascending=False).head(15))
display(errors.groupby('hour').agg(mean_abs_error=('abs_error','mean'), mean_signed_error=('error','mean'), count=('error','size')).sort_values('mean_abs_error', ascending=False))

# %% [markdown]
# ## 8. Feature importance and final interpretation

# %%
importance = permutation_importance(tuned_model, X_test, y_test, n_repeats=3, random_state=RANDOM_STATE, scoring='neg_root_mean_squared_error', n_jobs=-1)
importance_df = pd.DataFrame({'feature': X_test.columns, 'importance': importance.importances_mean}).sort_values('importance', ascending=False)
display(importance_df.head(20))
sns.barplot(data=importance_df.head(15), x='importance', y='feature', color='steelblue'); plt.title('Top permutation importances'); plt.tight_layout(); plt.show()
print('Workflow complete: compare the final holdout table, then use the error analysis to decide the next feature or model improvement.')
