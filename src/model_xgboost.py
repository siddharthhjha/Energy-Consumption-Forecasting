"""Train and evaluate a 24-hour direct XGBoost demand forecaster."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from xgboost import XGBRegressor
from config import FORECAST_HORIZON_HOURS, MODEL_DIR, OUTPUT_DIR, PROCESSED_DIR, RANDOM_STATE, TEST_START
from src.data_pipeline import load_merged_data
from src.evaluation import calculate_metrics, save_visualizations
from src.feature_engineering import build_features


def main() -> None:
    raw = load_merged_data()
    featured, feature_cols = build_features(raw, FORECAST_HORIZON_HOURS)
    featured.to_csv(PROCESSED_DIR / "processed_data.csv")
    usable = featured.dropna(subset=feature_cols + ["target_24h"])
    # Split by the target timestamp, ensuring no future labels cross the boundary.
    train = usable[usable.index + pd.Timedelta(hours=FORECAST_HORIZON_HOURS) < pd.Timestamp(TEST_START)]
    test = usable[usable.index + pd.Timedelta(hours=FORECAST_HORIZON_HOURS) >= pd.Timestamp(TEST_START)]
    model = XGBRegressor(n_estimators=900, learning_rate=0.035, max_depth=7, min_child_weight=5, subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0, objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=50)
    model.fit(train[feature_cols], train["target_24h"], eval_set=[(test[feature_cols], test["target_24h"])], verbose=False)
    prediction_index = test.index + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    predictions = pd.DataFrame({"actual": test["target_24h"].to_numpy(), "predicted": model.predict(test[feature_cols])}, index=prediction_index)
    predictions.index.name = "Datetime"
    metrics = calculate_metrics(predictions["actual"], predictions["predicted"])
    OUTPUT_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    baseline_metrics = calculate_metrics(test["target_24h"], test["PJME_MW"])
    pd.DataFrame([
        {"model": "XGBoost direct 24h", **metrics},
        {"model": "Daily seasonal naive", **baseline_metrics},
    ]).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv")
    importance = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    save_visualizations(predictions, importance, OUTPUT_DIR)
    model.save_model(MODEL_DIR / "xgboost_energy_model.json")
    print("24-hour direct forecast evaluation")
    for key, value in metrics.items(): print(f"{key}: {value:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Artifacts: {OUTPUT_DIR} and {MODEL_DIR}")


if __name__ == "__main__":
    main()
