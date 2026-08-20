"""Project configuration. Secrets are loaded from environment variables."""
from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT_DIR / "PJME_hourly.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
WEATHER_FILE = ROOT_DIR / "data" / "weather_data.csv"

LATITUDE = 39.95
LONGITUDE = -75.17
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
FORECAST_HORIZON_HOURS = 24
TEST_START = "2017-01-01"
RANDOM_STATE = 42
