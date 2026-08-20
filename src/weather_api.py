"""Weather retrieval and deterministic historical-weather fallback utilities."""
from __future__ import annotations

from pathlib import Path
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from config import LATITUDE, LONGITUDE, OPENWEATHER_API_KEY

WEATHER_COLUMNS = ["Temperature", "Humidity", "WindSpeed", "Pressure", "CloudCover"]


def fetch_current_weather(api_key: str | None = None) -> dict:
    """Fetch current conditions; useful for serving, not historical backfilling."""
    key = api_key or OPENWEATHER_API_KEY
    if not key:
        raise ValueError("OPENWEATHER_API_KEY is not configured.")
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": LATITUDE, "lon": LONGITUDE, "appid": key, "units": "metric"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "Datetime": pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h"),
        "Temperature": payload["main"]["temp"],
        "Humidity": payload["main"]["humidity"],
        "WindSpeed": payload["wind"]["speed"],
        "Pressure": payload["main"]["pressure"],
        "CloudCover": payload.get("clouds", {}).get("all", np.nan),
    }


def fetch_historical_weather(timestamps: Iterable[pd.Timestamp], api_key: str | None = None) -> pd.DataFrame:
    """Retrieve historical data using OpenWeather's Time Machine endpoint.

    The endpoint normally requires a paid historical subscription. Requests are
    deliberately rate-limited and the function raises a clear error when the
    account/API does not permit historical access.
    """
    key = api_key or OPENWEATHER_API_KEY
    if not key:
        raise ValueError("OPENWEATHER_API_KEY is not configured.")
    unique_days = pd.DatetimeIndex(pd.to_datetime(list(timestamps))).normalize().unique()
    rows: list[dict] = []
    for day in unique_days:
        response = requests.get(
            "https://api.openweathermap.org/data/3.0/onecall/timemachine",
            params={"lat": LATITUDE, "lon": LONGITUDE, "dt": int(day.timestamp()), "appid": key, "units": "metric"},
            timeout=30,
        )
        response.raise_for_status()
        for item in response.json().get("data", []):
            rows.append({
                "Datetime": pd.to_datetime(item["dt"], unit="s"),
                "Temperature": item.get("temp"), "Humidity": item.get("humidity"),
                "WindSpeed": item.get("wind_speed"), "Pressure": item.get("pressure"),
                "CloudCover": item.get("clouds"),
            })
        time.sleep(1)
    return pd.DataFrame(rows)


def make_climatology_weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create transparent, deterministic proxy weather when no historic feed exists.

    These values are seasonal climatology placeholders, never observations; this
    lets the pipeline run end-to-end while keeping the weather provenance explicit.
    """
    idx = pd.DatetimeIndex(index)
    annual = np.sin(2 * np.pi * (idx.dayofyear.to_numpy() - 172) / 365.25)
    daily = np.sin(2 * np.pi * (idx.hour.to_numpy() - 14) / 24)
    weather = pd.DataFrame(index=idx)
    weather["Temperature"] = 13 + 12 * annual + 3 * daily
    weather["Humidity"] = np.clip(67 - 14 * annual - 4 * daily, 20, 100)
    weather["WindSpeed"] = 4.2 + 1.2 * np.cos(2 * np.pi * idx.dayofyear.to_numpy() / 11)
    weather["Pressure"] = 1014 + 5 * np.cos(2 * np.pi * idx.dayofyear.to_numpy() / 18)
    weather["CloudCover"] = np.clip(55 + 25 * np.sin(2 * np.pi * idx.dayofyear.to_numpy() / 9), 0, 100)
    weather.index.name = "Datetime"
    return weather.reset_index()


def load_or_create_weather(index: pd.DatetimeIndex, path: Path) -> pd.DataFrame:
    """Load supplied weather, otherwise create and persist a labeled climatology proxy."""
    if path.exists():
        weather = pd.read_csv(path, parse_dates=["Datetime"])
        missing = set(WEATHER_COLUMNS) - set(weather.columns)
        if missing:
            raise ValueError(f"Weather file missing required columns: {sorted(missing)}")
        return weather
    weather = make_climatology_weather(index)
    weather["weather_source"] = "deterministic_climatology_proxy"
    path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(path, index=False)
    return weather
