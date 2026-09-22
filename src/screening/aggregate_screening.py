"""Aggregates results/screening/screening_results.csv into per-candidate,
per-testbed mean+-std (across seeds) tables for MODEL_SELECTION_REPORT.md."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "results" / "screening" / "screening_results.csv")

metrics = ["macro_f1", "balanced_accuracy", "minority_recall", "FPR", "PR_AUC", "ROC_AUC"]
agg = df.groupby(["candidate", "testbed"])[metrics].agg(["mean", "std"]).round(4)
print(agg.to_string())
agg.to_csv(ROOT / "results" / "screening" / "screening_summary.csv")

overall = df.groupby("candidate")[metrics].mean().round(4).sort_values("macro_f1", ascending=False)
print("\n=== Overall mean across all testbeds/seeds ===")
print(overall.to_string())
overall.to_csv(ROOT / "results" / "screening" / "screening_overall.csv")
