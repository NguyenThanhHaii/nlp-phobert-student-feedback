"""Merge Stage 6A baseline outputs and Stage 6B PhoBERT outputs.

Run from project root after copying 06b_* CSV files into reports/tables.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd()
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

baseline_results_path = TABLES_DIR / "06_noisy_evaluation_all_models.csv"
phobert_results_path = TABLES_DIR / "06b_phobert_noisy_evaluation_all.csv"

baseline_drop_path = TABLES_DIR / "06_robustness_drop_all_models.csv"
phobert_drop_path = TABLES_DIR / "06b_phobert_robustness_drop_all.csv"

missing = [
    path for path in [
        baseline_results_path,
        phobert_results_path,
        baseline_drop_path,
        phobert_drop_path,
    ]
    if not path.exists()
]

if missing:
    for path in missing:
        print("Missing:", path)
    raise FileNotFoundError("Missing required Stage 6A/6B result files.")

baseline_results = pd.read_csv(baseline_results_path)
phobert_results = pd.read_csv(phobert_results_path)

full_results = pd.concat(
    [baseline_results, phobert_results],
    ignore_index=True,
)

full_results_path = TABLES_DIR / "06_full_noisy_evaluation_all_models.csv"
full_results.to_csv(full_results_path, index=False, encoding="utf-8-sig")

baseline_drop = pd.read_csv(baseline_drop_path)
phobert_drop = pd.read_csv(phobert_drop_path)

full_drop = pd.concat(
    [baseline_drop, phobert_drop],
    ignore_index=True,
)

full_drop_path = TABLES_DIR / "06_full_robustness_drop_all_models.csv"
full_drop.to_csv(full_drop_path, index=False, encoding="utf-8-sig")

ranking = (
    full_drop[full_drop["evaluation_scope"] == "full"]
    .groupby(["task", "model_type", "model_name"])
    .agg(
        mean_absolute_macro_f1_drop=("absolute_macro_f1_drop", "mean"),
        max_absolute_macro_f1_drop=("absolute_macro_f1_drop", "max"),
        mean_relative_macro_f1_drop_pct=("relative_macro_f1_drop_pct", "mean"),
    )
    .reset_index()
    .sort_values(["task", "mean_absolute_macro_f1_drop"], ascending=[True, True])
)

ranking_path = TABLES_DIR / "06_full_model_robustness_ranking.csv"
ranking.to_csv(ranking_path, index=False, encoding="utf-8-sig")

print("Saved:", full_results_path)
print("Saved:", full_drop_path)
print("Saved:", ranking_path)
