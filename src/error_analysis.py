
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def load_noisy_files(noisy_dir: Path, pattern: str = "04_test_*.csv") -> dict[str, pd.DataFrame]:
    noisy_dir = Path(noisy_dir)
    files = sorted(noisy_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No noisy files found in {noisy_dir} with pattern {pattern}")

    frames = {}
    for path in files:
        df = pd.read_csv(path)
        if "noise_type" in df.columns and len(df) > 0:
            noise_type = str(df["noise_type"].iloc[0])
        else:
            noise_type = path.stem.replace("04_test_", "")
        frames[noise_type] = df
    return frames


def build_text_lookup(clean_test_df: pd.DataFrame, noisy_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for _, row in clean_test_df.iterrows():
        rows.append({
            "id": row["id"],
            "noise_type": "clean",
            "original_text": row["text"],
            "noisy_text": row["text"],
            "text": row["text"],
            "severity_lookup": "clean",
            "changed_token_ratio_lookup": 0.0,
            "is_changed_lookup": True,
        })

    for noise_type, noisy_df in noisy_frames.items():
        required = ["id", "original_text", "noisy_text", "severity", "changed_token_ratio", "is_changed"]
        missing = [col for col in required if col not in noisy_df.columns]
        if missing:
            raise KeyError(f"Missing noisy columns for {noise_type}: {missing}")

        temp = noisy_df[required].copy()
        temp["noise_type"] = noise_type
        temp["text"] = temp["noisy_text"]
        temp = temp.rename(columns={
            "severity": "severity_lookup",
            "changed_token_ratio": "changed_token_ratio_lookup",
            "is_changed": "is_changed_lookup",
        })

        rows.extend(temp[[
            "id", "noise_type", "original_text", "noisy_text", "text",
            "severity_lookup", "changed_token_ratio_lookup", "is_changed_lookup",
        ]].to_dict("records"))

    return pd.DataFrame(rows)


def load_predictions(
    baseline_predictions_path: Path,
    phobert_predictions_light_path: Path,
    text_lookup_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    baseline_predictions_path = Path(baseline_predictions_path)
    phobert_predictions_light_path = Path(phobert_predictions_light_path)

    if not baseline_predictions_path.exists():
        raise FileNotFoundError(f"Missing baseline predictions: {baseline_predictions_path}")
    if not phobert_predictions_light_path.exists():
        raise FileNotFoundError(f"Missing PhoBERT predictions: {phobert_predictions_light_path}")

    baseline_df = pd.read_csv(baseline_predictions_path)
    frames.append(baseline_df)

    phobert_df = pd.read_csv(phobert_predictions_light_path)
    phobert_df = phobert_df.merge(
        text_lookup_df.drop_duplicates(["id", "noise_type"]),
        on=["id", "noise_type"],
        how="left",
    )

    for col in ["severity", "changed_token_ratio", "is_changed"]:
        lookup_col = f"{col}_lookup"
        if col not in phobert_df.columns and lookup_col in phobert_df.columns:
            phobert_df[col] = phobert_df[lookup_col]
        elif col in phobert_df.columns and lookup_col in phobert_df.columns:
            phobert_df[col] = phobert_df[col].where(phobert_df[col].notna(), phobert_df[lookup_col])

    frames.append(phobert_df)

    predictions = pd.concat(frames, ignore_index=True)

    required = [
        "task", "model_type", "model_name", "noise_type", "evaluation_scope",
        "id", "true_label", "pred_label", "is_correct"
    ]
    missing = [col for col in required if col not in predictions.columns]
    if missing:
        raise KeyError(f"Missing prediction columns: {missing}")

    predictions["true_label"] = predictions["true_label"].astype(str).str.lower()
    predictions["pred_label"] = predictions["pred_label"].astype(str).str.lower()
    predictions["is_correct"] = predictions["is_correct"].astype(bool)

    return predictions


def add_tokenization_features(predictions_df: pd.DataFrame, tokenization_analysis_path: Path) -> pd.DataFrame:
    tokenization_analysis_path = Path(tokenization_analysis_path)
    if not tokenization_analysis_path.exists():
        print(f"[WARNING] Missing tokenization analysis: {tokenization_analysis_path}")
        return predictions_df

    token_df = pd.read_csv(tokenization_analysis_path)

    keep_cols = [
        "id", "noise_type",
        "num_subwords_clean", "num_subwords_noisy", "subword_delta",
        "subword_inflation_ratio", "subword_change_flag",
        "segmentation_change_flag", "segment_delta",
        "segmentation_jaccard_distance", "subword_jaccard_distance",
    ]
    keep_cols = [col for col in keep_cols if col in token_df.columns]

    return predictions_df.merge(
        token_df[keep_cols].drop_duplicates(["id", "noise_type"]),
        on=["id", "noise_type"],
        how="left",
    )


def summarize_error_rates(predictions_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        predictions_df
        .groupby(["task", "model_type", "model_name", "noise_type", "evaluation_scope"])
        .agg(
            num_samples=("id", "count"),
            num_errors=("is_correct", lambda s: int((~s).sum())),
            accuracy=("is_correct", "mean"),
            mean_changed_token_ratio=("changed_token_ratio", "mean"),
        )
        .reset_index()
    )
    summary["error_rate"] = 1.0 - summary["accuracy"]
    return summary


def summarize_confusion_pairs(predictions_df: pd.DataFrame) -> pd.DataFrame:
    error_df = predictions_df[~predictions_df["is_correct"]].copy()
    if error_df.empty:
        return pd.DataFrame()

    return (
        error_df
        .groupby([
            "task", "model_type", "model_name", "noise_type", "evaluation_scope",
            "true_label", "pred_label"
        ])
        .size()
        .reset_index(name="num_errors")
        .sort_values(["task", "model_name", "noise_type", "num_errors"], ascending=[True, True, True, False])
    )


def select_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "task", "model_type", "model_name", "noise_type", "evaluation_scope",
        "id", "true_label", "pred_label", "severity", "changed_token_ratio",
        "subword_delta", "subword_inflation_ratio",
        "original_text", "noisy_text", "text",
    ]
    return df[[col for col in preferred if col in df.columns]].copy()


def sample_error_rows(
    predictions_df: pd.DataFrame,
    top_n_per_group: int = 30,
    exclude_majority: bool = True,
) -> pd.DataFrame:
    error_df = predictions_df[~predictions_df["is_correct"]].copy()
    if exclude_majority:
        error_df = error_df[error_df["model_name"] != "majority_class"].copy()

    if error_df.empty:
        return error_df

    sort_cols = []
    ascending = []
    for col in ["subword_delta", "subword_inflation_ratio", "changed_token_ratio"]:
        if col in error_df.columns:
            sort_cols.append(col)
            ascending.append(False)

    if sort_cols:
        error_df = error_df.sort_values(sort_cols, ascending=ascending)

    return (
        error_df
        .groupby(["task", "model_type", "model_name", "noise_type", "evaluation_scope"], group_keys=False)
        .head(top_n_per_group)
        .reset_index(drop=True)
    )


def minority_class_errors(
    predictions_df: pd.DataFrame,
    class_map: dict[str, list[str]],
    exclude_majority: bool = True,
) -> pd.DataFrame:
    error_df = predictions_df[~predictions_df["is_correct"]].copy()
    if exclude_majority:
        error_df = error_df[error_df["model_name"] != "majority_class"].copy()

    frames = []
    for task, labels in class_map.items():
        labels = [str(label).lower() for label in labels]
        task_df = error_df[(error_df["task"] == task) & (error_df["true_label"].isin(labels))].copy()
        frames.append(task_df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def no_accent_errors(predictions_df: pd.DataFrame, exclude_majority: bool = True) -> pd.DataFrame:
    error_df = predictions_df[
        (~predictions_df["is_correct"]) & (predictions_df["noise_type"] == "no_accent")
    ].copy()

    if exclude_majority:
        error_df = error_df[error_df["model_name"] != "majority_class"].copy()
    return error_df


def tokenization_error_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if "subword_inflation_ratio" not in predictions_df.columns:
        return pd.DataFrame()

    df = predictions_df[predictions_df["noise_type"] != "clean"].copy()
    if df.empty:
        return pd.DataFrame()

    bins = [0, 0.95, 1.0, 1.05, 1.15, 1.30, np.inf]
    labels = ["<0.95", "0.95-1.00", "1.00-1.05", "1.05-1.15", "1.15-1.30", ">1.30"]

    df["subword_inflation_bin"] = pd.cut(
        df["subword_inflation_ratio"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    summary = (
        df
        .groupby([
            "task", "model_type", "model_name", "noise_type",
            "evaluation_scope", "subword_inflation_bin"
        ], observed=True)
        .agg(
            num_samples=("id", "count"),
            num_errors=("is_correct", lambda s: int((~s).sum())),
            accuracy=("is_correct", "mean"),
            mean_subword_delta=("subword_delta", "mean"),
            mean_changed_token_ratio=("changed_token_ratio", "mean"),
        )
        .reset_index()
    )
    summary["error_rate"] = 1.0 - summary["accuracy"]
    return summary
