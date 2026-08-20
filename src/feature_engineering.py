"""Feature creation for leakage-safe, direct multi-step demand forecasting."""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]


def build_features(data: pd.DataFrame, horizon: int = 24) -> tuple[pd.DataFrame, list[str]]:
    df = data.copy()
    # Calendar and proxy weather must describe the timestamp being forecast,
    # rather than the origin timestamp.
    idx = df.index + pd.Timedelta(hours=horizon)
    df["hour"] = idx.hour
    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["quarter"] = idx.quarter
    df["year"] = idx.year
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    for lag in LAGS:
        df[f"lag_{lag}h"] = df["PJME_MW"].shift(lag)
    # shift first: the origin's feature cannot include the target-period value.
    df["rolling_mean_24h"] = df["PJME_MW"].shift(1).rolling(24).mean()
    df["rolling_std_24h"] = df["PJME_MW"].shift(1).rolling(24).std()
    # A direct model trained at t forecasts the demand at t + horizon.
    df["target_24h"] = df["PJME_MW"].shift(-horizon)
    for column in ["Temperature", "Humidity", "WindSpeed", "Pressure", "CloudCover"]:
        # In production this should be the weather forecast valid at t+horizon.
        df[column] = df[column].shift(-horizon)
    features = [
        "hour_sin", "hour_cos", "day_sin", "day_cos", "month", "quarter", "year",
        *[f"lag_{lag}h" for lag in LAGS], "rolling_mean_24h", "rolling_std_24h",
        "Temperature", "Humidity", "WindSpeed", "Pressure", "CloudCover",
    ]
    return df, features
