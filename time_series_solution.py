"""Leakage-safe time-series solution for the Appliances Energy dataset.

Run from this folder with:
    python3 time_series_solution.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")


def make_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create features using only information available at prediction time."""
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.sort_values("date").reset_index(drop=True)

    # Calendar features
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["hour"] = data["date"].dt.hour
    data["minute"] = data["date"].dt.minute
    data["time_of_day"] = data["hour"] * 6 + data["minute"] // 10
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    # Cyclical encoding preserves the fact that 23:50 and 00:00 are close.
    data["hour_sin"] = np.sin(2 * np.pi * data["time_of_day"] / 144)
    data["hour_cos"] = np.cos(2 * np.pi * data["time_of_day"] / 144)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)

    # 10-minute data: 6 rows/hour, 144 rows/day, 1008 rows/week.
    target = data["Appliances"]
    for lag in [1, 2, 3, 6, 12, 18, 36, 72, 144, 288, 1008]:
        data[f"appliances_lag_{lag}"] = target.shift(lag)
    for window in [6, 18, 36, 144]:
        past = target.shift(1)
        data[f"appliances_roll_mean_{window}"] = past.rolling(window).mean()
        data[f"appliances_roll_std_{window}"] = past.rolling(window).std()

    # Date is no longer needed; rows with unavailable history are removed.
    data = data.drop(columns=["date", "Appliances", "rv2"], errors="ignore")
    return data, target.loc[data.index]


def main() -> None:
    path = Path(__file__).with_name("energydata_complete.csv")
    raw = pd.read_csv(path)
    X, y = make_features(raw)

    # Keep chronological order. The last 20% is a genuinely future holdout.
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    numeric = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]
    preprocess = ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
            ]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )

    models = {
        "Naive previous value": None,
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42,
        ),
    }

    rows = []
    # Naive forecast uses the latest known observation and is a key baseline.
    naive_pred = X_test["appliances_lag_1"].to_numpy()
    rows.append({
        "model": "Naive previous value",
        "r2": r2_score(y_test, naive_pred),
        "mae": mean_absolute_error(y_test, naive_pred),
        "rmse": mean_squared_error(y_test, naive_pred) ** 0.5,
    })

    for name, estimator in list(models.items())[1:]:
        model = Pipeline([("preprocess", preprocess), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "r2": r2_score(y_test, pred),
            "mae": mean_absolute_error(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
        })

    result = pd.DataFrame(rows).sort_values("rmse")
    print(f"Rows: {len(raw):,} | Train: {len(X_train):,} | Future test: {len(X_test):,}")
    print("\nFinal chronological holdout scores (higher R2, lower MAE/RMSE is better):")
    print(result.to_string(index=False, formatters={
        "r2": "{:.4f}".format,
        "mae": "{:.2f}".format,
        "rmse": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
