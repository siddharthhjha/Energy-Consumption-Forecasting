# Energy Consumption Forecasting with Weather Integration

An XGBoost system that forecasts PJM East electricity demand 24 hours ahead from calendar, lag, rolling, and weather features.

## Quick start

```powershell
pip install -r requirements.txt
python -m src.model_xgboost
python -m src.eda
streamlit run app.py
```

The model run reads `PJME_hourly.csv`, writes the processed data to `data/processed/`, stores the model at `models/xgboost_energy_model.json`, and saves metrics, predictions, feature importance, benchmark comparison, and plots under `outputs/`. The EDA command writes a data summary and overview chart.

The Streamlit dashboard in `app.py` presents the project requirements, interactive EDA, weather provenance, engineered features, evaluation results, ER-style data lineage, and file structure.

## Forecast design

This is a direct 24-hour forecast: at origin time `t`, the target is `PJME_MW(t + 24h)`. Lag and rolling features are based only on data available at `t` or earlier, and the chronological split is performed using the target timestamp. This avoids evaluating a purported 24-hour forecast with future actual load values.

## Weather data

`src/weather_api.py` supports the OpenWeather current-weather endpoint for serving and includes a historical Time Machine client (which needs an appropriate OpenWeather subscription). The PRD's listed `/data/2.5/weather` endpoint is current-only and cannot backfill the 2002–2018 PJM period.

For an end-to-end reproducible local run, the pipeline creates `data/weather_data.csv` with a clearly labeled deterministic seasonal climatology proxy if a real historical file has not been supplied. Replace that file with observed historical hourly weather columns (`Datetime`, `Temperature`, `Humidity`, `WindSpeed`, `Pressure`, `CloudCover`) before treating weather importance or accuracy as production results. Keep API keys in `.env`, never source code.
