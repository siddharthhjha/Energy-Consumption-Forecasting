"""Metrics and project visualizations."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(actual, predicted) -> dict[str, float]:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"MAE_MW": mean_absolute_error(actual, predicted), "RMSE_MW": np.sqrt(mean_squared_error(actual, predicted)), "MAPE_pct": np.mean(np.abs((actual - predicted) / actual)) * 100, "R2": r2_score(actual, predicted)}


def save_visualizations(predictions: pd.DataFrame, importance: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    sample = predictions.iloc[:24 * 14]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample.index, sample["actual"], label="Actual", linewidth=1.5)
    ax.plot(sample.index, sample["predicted"], label="24h forecast", linewidth=1.3)
    ax.set(title="PJM East: Actual vs 24-hour-ahead Forecast", ylabel="MW", xlabel="Target timestamp")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output_dir / "predictions_plot.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=importance.head(15), x="importance", y="feature", ax=ax, color="#2678b2")
    ax.set(title="XGBoost Feature Importance (top 15)", xlabel="Gain importance", ylabel="")
    fig.tight_layout(); fig.savefig(output_dir / "feature_importance.png", dpi=150); plt.close(fig)
    residuals = predictions["actual"] - predictions["predicted"]
    fig, ax = plt.subplots(figsize=(10, 5)); sns.histplot(residuals, kde=True, ax=ax)
    ax.set(title="Forecast Residual Distribution", xlabel="Actual − predicted (MW)")
    fig.tight_layout(); fig.savefig(output_dir / "residuals_plot.png", dpi=150); plt.close(fig)
