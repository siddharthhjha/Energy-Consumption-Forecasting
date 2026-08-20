"""Create compact exploratory outputs for the PJM source dataset."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import OUTPUT_DIR
from src.data_pipeline import load_energy_data


def main() -> None:
    data = load_energy_data()
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = data.describe().T
    summary["missing_values"] = data.isna().sum()
    summary.to_csv(OUTPUT_DIR / "data_summary_statistics.csv")
    print(f"Records: {len(data):,}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(summary.round(2).to_string())
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data["PJME_MW"].plot(ax=axes[0], linewidth=.35, color="#276fbf")
    axes[0].set(title="PJM East hourly demand history", ylabel="MW", xlabel="")
    profile = data.assign(hour=data.index.hour).groupby("hour")["PJME_MW"].mean()
    profile.plot(ax=axes[1], marker="o", color="#e07028")
    axes[1].set(title="Average demand by hour", xlabel="Hour of day", ylabel="MW")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "eda_overview.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
