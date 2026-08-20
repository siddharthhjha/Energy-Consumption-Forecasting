"""Load, validate, regularize, and merge PJM energy/weather data."""
from __future__ import annotations
import pandas as pd
from config import DATA_FILE, WEATHER_FILE
from src.weather_api import load_or_create_weather


def load_energy_data(path=DATA_FILE) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["Datetime"])
    required = {"Datetime", "PJME_MW"}
    if missing := required - set(data.columns):
        raise ValueError(f"Energy file is missing: {sorted(missing)}")
    data = data.sort_values("Datetime").drop_duplicates("Datetime", keep="last").set_index("Datetime")
    data = data.asfreq("h")
    data["PJME_MW"] = data["PJME_MW"].interpolate(method="time", limit=3).ffill().bfill()
    return data


def load_merged_data() -> pd.DataFrame:
    energy = load_energy_data()
    weather = load_or_create_weather(energy.index, WEATHER_FILE)
    weather = weather.drop_duplicates("Datetime", keep="last").set_index("Datetime").sort_index()
    return energy.join(weather.drop(columns=["weather_source"], errors="ignore"), how="left").ffill().bfill()
