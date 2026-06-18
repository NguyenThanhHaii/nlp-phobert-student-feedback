
"""Stage 8 optional normalization utilities.

The normalizer is intentionally simple and rule-based. It is not meant to be a
full Vietnamese spell checker. The goal is to test whether lightweight input
normalization can recover some performance on controlled noisy test sets.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, f1_score


def _word_boundary_pattern(term: str) -> re.Pattern:
    """Build a case-insensitive regex for a token/phrase with non-word boundaries."""
    escaped = re.escape(str(term))
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def apply_mapping_replacements(text: str, replacements: dict[str, str], rule_name: str) -> tuple[str, list[dict]]:
    """Apply token/phrase replacements and record spans."""
    current = str(text)
    spans: list[dict] = []

    # Longest first avoids replacing subphrases too early.
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = _word_boundary_pattern(source)
        matches = list(pattern.finditer(current))

        if not matches:
            continue

        before = current
        current = pattern.sub(str(target), current)

        if before != current:
            for match in matches:
                spans.append({
                    "rule": rule_name,
                    "source": source,
                    "target": target,
                    "start_char": match.start(),
                    "end_char": match.end(),
                    "matched_text": match.group(0),
                })

    return current, spans


def reduce_elongation(text: str, min_repeat: int = 3) -> tuple[str, list[dict]]:
    """Reduce repeated characters, e.g. tốttt -> tốt, hayyy -> hay."""
    current = str(text)
    spans: list[dict] = []

    pattern = re.compile(rf"(.)\1{{{min_repeat - 1},}}", flags=re.IGNORECASE)

    for match in pattern.finditer(current):
        repeated = match.group(0)
        replacement = match.group(1)
        spans.append({
            "rule": "reduce_elongation",
            "source": repeated,
            "target": replacement,
            "start_char": match.start(),
            "end_char": match.end(),
            "matched_text": repeated,
        })

    normalized = pattern.sub(lambda m: m.group(1), current)
    return normalized, spans


def normalize_text(text: str, config: dict[str, Any]) -> tuple[str, list[dict]]:
    """Normalize one text with configured rules."""
    current = str(text)
    all_spans: list[dict] = []

    norm_cfg = config.get("normalization", {})
    apply_order = norm_cfg.get("apply_order", [])

    for rule_name in apply_order:
        rule_cfg = norm_cfg.get(rule_name, {})
        if not bool(rule_cfg.get("enabled", True)):
            continue

        if rule_name == "reduce_elongation":
            current, spans = reduce_elongation(
                current,
                min_repeat=int(rule_cfg.get("min_repeat", 3)),
            )

        elif rule_name in {"domain_abbreviation", "teencode", "no_accent_phrase", "common_typo"}:
            current, spans = apply_mapping_replacements(
                current,
                replacements=rule_cfg.get("replacements", {}),
                rule_name=rule_name,
            )

        else:
            spans = []

        all_spans.extend(spans)

    return current, all_spans


def normalize_noisy_dataframe(noisy_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add normalized_text and normalization metadata to one noisy dataframe."""
    required = [
        "id",
        "original_text",
        "noisy_text",
        "sentiment_label",
        "topic_label",
        "noise_type",
        "is_changed",
        "changed_token_ratio",
        "severity",
    ]
    missing = [col for col in required if col not in noisy_df.columns]
    if missing:
        raise KeyError(f"Missing noisy dataframe columns: {missing}")

    rows = []
    for _, row in noisy_df.iterrows():
        noisy_text = str(row["noisy_text"])
        normalized_text, spans = normalize_text(noisy_text, config)

        out = row.to_dict()
        out["normalized_text"] = normalized_text
        out["normalization_spans"] = json.dumps(spans, ensure_ascii=False)
        out["num_normalization_replacements"] = len(spans)
        out["normalization_changed"] = normalized_text != noisy_text
        rows.append(out)

    return pd.DataFrame(rows)


def load_label_mapping(path: Path) -> dict[str, dict[str, str]]:
    """Load Stage 1 label mapping."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing label mapping: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        task: {str(k): str(v).lower() for k, v in raw[task]["id_to_name"].items()}
        for task in ["sentiment", "topic"]
    }


def get_label_names(label_mapping: dict[str, dict[str, str]], task: str) -> list[str]:
    """Return labels in numeric-id order."""
    return [
        label_mapping[task][key]
        for key in sorted(label_mapping[task].keys(), key=lambda value: int(value))
    ]


def normalize_predictions(predictions, task: str, label_mapping: dict[str, dict[str, str]]) -> list[str]:
    """Normalize integer/string predictions to label names."""
    id_to_name = label_mapping[task]
    out = []
    for pred in predictions:
        if isinstance(pred, (int, np.integer)):
            out.append(id_to_name[str(int(pred))])
        else:
            pred_str = str(pred)
            if pred_str in id_to_name:
                out.append(id_to_name[pred_str])
            else:
                out.append(pred_str.lower())
    return out


def discover_baseline_models(task: str, baseline_root: Path, fallback_roots: list[Path] | None = None) -> list[dict[str, Any]]:
    """Discover baseline joblib models."""
    candidate_dirs = [Path(baseline_root) / task]

    for root in fallback_roots or []:
        root = Path(root)
        if root == Path(baseline_root):
            continue
        candidate_dirs.append(root / task)

    specs = []
    for model_dir in candidate_dirs:
        if not model_dir.exists():
            continue

        for path in sorted(model_dir.glob("*.joblib")):
            specs.append({
                "task": task,
                "model_type": "baseline",
                "model_name": path.stem,
                "path": path,
            })

        if specs:
            break

    return specs


def prepare_eval_frame(noisy_df: pd.DataFrame, task: str, text_column: str, input_variant: str, evaluation_scope: str) -> pd.DataFrame:
    """Prepare evaluation dataframe."""
    label_col = f"{task}_label"

    df = noisy_df.copy()

    if evaluation_scope == "changed_only":
        df = df[df["is_changed"].astype(bool)].copy()
    elif evaluation_scope != "full":
        raise ValueError(f"Unsupported evaluation_scope: {evaluation_scope}")

    return pd.DataFrame({
        "id": df["id"],
        "text": df[text_column].astype(str),
        "true_label": df[label_col].astype(str).str.lower(),
        "task": task,
        "noise_type": df["noise_type"].astype(str),
        "input_variant": input_variant,
        "evaluation_scope": evaluation_scope,
        "is_changed": df["is_changed"].astype(bool),
        "normalization_changed": df.get("normalization_changed", False),
        "changed_token_ratio": df["changed_token_ratio"],
        "num_normalization_replacements": df.get("num_normalization_replacements", 0),
    })


def evaluate_sklearn_model(model_path: Path, eval_df: pd.DataFrame, task: str, label_mapping: dict[str, dict[str, str]]) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate one sklearn/joblib model on one eval dataframe."""
    model = joblib.load(model_path)
    labels = get_label_names(label_mapping, task)

    texts = eval_df["text"].astype(str).tolist()
    y_true = eval_df["true_label"].astype(str).str.lower().tolist()
    y_pred = normalize_predictions(model.predict(texts), task, label_mapping)

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    summary = {
        "task": task,
        "model_type": "baseline",
        "model_name": Path(model_path).stem,
        "model_path": str(model_path),
        "noise_type": str(eval_df["noise_type"].iloc[0]),
        "input_variant": str(eval_df["input_variant"].iloc[0]),
        "evaluation_scope": str(eval_df["evaluation_scope"].iloc[0]),
        "num_eval_samples": len(eval_df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    pred_df = eval_df.copy()
    pred_df["model_type"] = "baseline"
    pred_df["model_name"] = Path(model_path).stem
    pred_df["pred_label"] = y_pred
    pred_df["is_correct"] = pred_df["true_label"] == pred_df["pred_label"]

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    report_rows = []
    for label, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        report_rows.append({
            "task": task,
            "model_type": "baseline",
            "model_name": Path(model_path).stem,
            "noise_type": str(eval_df["noise_type"].iloc[0]),
            "input_variant": str(eval_df["input_variant"].iloc[0]),
            "evaluation_scope": str(eval_df["evaluation_scope"].iloc[0]),
            "label": label,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1-score"),
            "support": metrics.get("support"),
        })

    return summary, pred_df, report_rows


def compute_normalization_improvement(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compare normalized vs noisy scores for same task/model/noise/scope."""
    key_cols = ["task", "model_type", "model_name", "noise_type", "evaluation_scope"]

    noisy_df = results_df[results_df["input_variant"] == "noisy"].copy()
    norm_df = results_df[results_df["input_variant"] == "normalized"].copy()

    merged = noisy_df.merge(
        norm_df,
        on=key_cols,
        suffixes=("_noisy", "_normalized"),
    )

    rows = []
    for _, row in merged.iterrows():
        rows.append({
            **{col: row[col] for col in key_cols},
            "noisy_macro_f1": row["macro_f1_noisy"],
            "normalized_macro_f1": row["macro_f1_normalized"],
            "macro_f1_improvement": row["macro_f1_normalized"] - row["macro_f1_noisy"],
            "noisy_accuracy": row["accuracy_noisy"],
            "normalized_accuracy": row["accuracy_normalized"],
            "accuracy_improvement": row["accuracy_normalized"] - row["accuracy_noisy"],
            "num_eval_samples": row["num_eval_samples_noisy"],
        })

    return pd.DataFrame(rows)


def summarize_normalized_generation(normalized_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize normalization generation effects."""
    rows = []
    for noise_type, df in normalized_frames.items():
        rows.append({
            "noise_type": noise_type,
            "num_rows": len(df),
            "num_noisy_changed_rows": int(df["is_changed"].sum()),
            "num_normalization_changed_rows": int(df["normalization_changed"].sum()),
            "normalization_changed_percentage": float(df["normalization_changed"].mean() * 100),
            "mean_num_normalization_replacements": float(df["num_normalization_replacements"].mean()),
            "max_num_normalization_replacements": int(df["num_normalization_replacements"].max()),
        })
    return pd.DataFrame(rows)
