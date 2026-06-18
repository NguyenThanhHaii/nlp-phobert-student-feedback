"""Stage 6 utilities for evaluating clean/noisy robustness.

This module evaluates:

- Majority Class baseline
- TF-IDF word-level + Linear SVM
- TF-IDF char-level + Linear SVM
- PhoBERT-base fine-tuned models

on clean and controlled noisy test sets.

The module is intentionally file-system based because previous stages save model
artifacts and reports locally.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, f1_score


@dataclass
class ModelSpec:
    """Model descriptor used by the evaluation loop."""

    task: str
    model_type: str
    model_name: str
    path: Path


def slugify(value: str) -> str:
    """Create a filesystem-safe slug."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def load_label_mapping(path: Path) -> dict[str, dict[str, str]]:
    """Load Stage 1 label mapping and normalize keys."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing label mapping: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapping = {}

    for task in ["sentiment", "topic"]:
        if task not in raw:
            raise KeyError(f"Missing task in label mapping: {task}")

        id_to_name = raw[task]["id_to_name"]
        mapping[task] = {
            str(key): str(value)
            for key, value in id_to_name.items()
        }

    return mapping


def get_label_names(label_mapping: dict[str, dict[str, str]], task: str) -> list[str]:
    """Return label names in numeric-id order."""
    id_to_name = label_mapping[task]
    return [
        id_to_name[key]
        for key in sorted(id_to_name.keys(), key=lambda value: int(value))
    ]


def normalize_predicted_labels(predictions: Any, task: str, label_mapping: dict[str, dict[str, str]]) -> list[str]:
    """Normalize predictions to label strings."""
    id_to_name = label_mapping[task]
    normalized = []

    for pred in predictions:
        if isinstance(pred, (np.integer, int)):
            normalized.append(id_to_name[str(int(pred))])
            continue

        pred_str = str(pred)

        if pred_str in id_to_name:
            normalized.append(id_to_name[pred_str])
        else:
            normalized.append(pred_str.lower())

    return normalized


def load_noisy_files(noisy_dir: Path, pattern: str = "04_test_*.csv") -> dict[str, pd.DataFrame]:
    """Load Stage 4 noisy test files."""
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


def prepare_clean_eval_frame(clean_df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Create a clean evaluation dataframe with a common schema."""
    label_col = f"{task}_label"
    required = ["id", "text", label_col]
    missing = [col for col in required if col not in clean_df.columns]

    if missing:
        raise KeyError(f"Missing clean test columns for {task}: {missing}")

    out = pd.DataFrame({
        "id": clean_df["id"],
        "text": clean_df["text"].astype(str),
        "true_label": clean_df[label_col].astype(str).str.lower(),
        "noise_type": "clean",
        "evaluation_scope": "full",
        "is_changed": True,
        "changed_token_ratio": 0.0,
        "severity": "clean",
        "original_text": clean_df["text"].astype(str),
        "noisy_text": clean_df["text"].astype(str),
    })

    return out


def prepare_noisy_eval_frame(noisy_df: pd.DataFrame, task: str, evaluation_scope: str = "full") -> pd.DataFrame:
    """Create a noisy evaluation dataframe with a common schema."""
    label_col = f"{task}_label"
    required = [
        "id",
        "original_text",
        "noisy_text",
        label_col,
        "noise_type",
        "is_changed",
        "changed_token_ratio",
        "severity",
    ]
    missing = [col for col in required if col not in noisy_df.columns]

    if missing:
        raise KeyError(f"Missing noisy columns for {task}: {missing}")

    df = noisy_df.copy()

    if evaluation_scope == "changed_only":
        df = df[df["is_changed"].astype(bool)].copy()
    elif evaluation_scope != "full":
        raise ValueError(f"Unsupported evaluation scope: {evaluation_scope}")

    out = pd.DataFrame({
        "id": df["id"],
        "text": df["noisy_text"].astype(str),
        "true_label": df[label_col].astype(str).str.lower(),
        "noise_type": df["noise_type"].astype(str),
        "evaluation_scope": evaluation_scope,
        "is_changed": df["is_changed"].astype(bool),
        "changed_token_ratio": df["changed_token_ratio"],
        "severity": df["severity"].astype(str),
        "original_text": df["original_text"].astype(str),
        "noisy_text": df["noisy_text"].astype(str),
    })

    return out


def compute_summary_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
) -> dict[str, float]:
    """Compute summary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
    }


def classification_report_rows(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    task: str,
    model_type: str,
    model_name: str,
    noise_type: str,
    evaluation_scope: str,
) -> list[dict[str, Any]]:
    """Flatten sklearn classification_report output into rows."""
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for label_name, metrics in report.items():
        if not isinstance(metrics, dict):
            continue

        row = {
            "task": task,
            "model_type": model_type,
            "model_name": model_name,
            "noise_type": noise_type,
            "evaluation_scope": evaluation_scope,
            "label": label_name,
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1-score"),
            "support": metrics.get("support"),
        }
        rows.append(row)

    return rows


def discover_baseline_models(
    task: str,
    baseline_root: Path,
    fallback_roots: list[Path] | None = None,
) -> list[ModelSpec]:
    """Discover baseline sklearn joblib models.

    Preferred structure:
        models/baselines/<task>/*.joblib

    Fallback structure:
        models/<task>/*.joblib
    """
    baseline_root = Path(baseline_root)
    candidate_dirs = [baseline_root / task]

    for root in fallback_roots or []:
        root = Path(root)

        # Avoid duplicating preferred root.
        if root == baseline_root:
            continue

        candidate_dirs.append(root / task)

    specs = []

    for model_dir in candidate_dirs:
        if not model_dir.exists():
            continue

        for path in sorted(model_dir.glob("*.joblib")):
            model_name = path.stem
            specs.append(
                ModelSpec(
                    task=task,
                    model_type="baseline",
                    model_name=model_name,
                    path=path,
                )
            )

        if specs:
            break

    return specs


def discover_phobert_model(task: str, phobert_root: Path) -> ModelSpec | None:
    """Discover PhoBERT best_model directory for a task."""
    candidate = Path(phobert_root) / task / "best_model"

    if not candidate.exists():
        return None

    return ModelSpec(
        task=task,
        model_type="phobert",
        model_name="phobert_base",
        path=candidate,
    )


def predict_sklearn_model(model_path: Path, texts: list[str], task: str, label_mapping: dict[str, dict[str, str]]) -> list[str]:
    """Load and predict with a sklearn/joblib model."""
    model = joblib.load(model_path)
    predictions = model.predict(texts)
    return normalize_predicted_labels(predictions, task=task, label_mapping=label_mapping)


def resolve_device(requested_device: str = "auto"):
    """Resolve torch device."""
    import torch

    requested_device = str(requested_device).strip().lower()

    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(requested_device)


def get_phobert_id2label(model, task: str, label_mapping: dict[str, dict[str, str]]) -> dict[int, str]:
    """Return id-to-label mapping for a PhoBERT model."""
    config_mapping = getattr(model.config, "id2label", None)

    if isinstance(config_mapping, dict):
        normalized = {}
        valid = True

        for key, value in config_mapping.items():
            try:
                key_int = int(key)
            except Exception:
                valid = False
                break

            value_str = str(value).lower()

            if value_str.startswith("label_"):
                valid = False
                break

            normalized[key_int] = value_str

        if valid and normalized:
            return normalized

    # Fallback to Stage 1 mapping.
    return {
        int(key): str(value).lower()
        for key, value in label_mapping[task].items()
    }


def predict_phobert_model(
    model_path: Path,
    texts: list[str],
    task: str,
    label_mapping: dict[str, dict[str, str]],
    fallback_model_name: str = "vinai/phobert-base",
    max_length: int = 128,
    batch_size: int = 32,
    device: str = "auto",
) -> list[str]:
    """Load and predict with a fine-tuned PhoBERT model."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(model_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=False)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    torch_device = resolve_device(device)
    model.to(torch_device)
    model.eval()

    id2label = get_phobert_id2label(model, task=task, label_mapping=label_mapping)

    encodings = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    dataset = TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
    )

    dataloader = DataLoader(dataset, batch_size=batch_size)

    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to(torch_device)
            attention_mask = batch[1].to(torch_device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            pred_ids = outputs.logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            all_predictions.extend([id2label[int(pred_id)] for pred_id in pred_ids])

    return all_predictions


def evaluate_model_on_frame(
    model_spec: ModelSpec,
    eval_df: pd.DataFrame,
    label_mapping: dict[str, dict[str, str]],
    phobert_model_name: str = "vinai/phobert-base",
    max_length: int = 128,
    phobert_batch_size: int = 32,
    device: str = "auto",
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Evaluate one model on one evaluation dataframe."""
    task = model_spec.task
    labels = get_label_names(label_mapping, task)
    texts = eval_df["text"].astype(str).tolist()
    y_true = eval_df["true_label"].astype(str).str.lower().tolist()

    if model_spec.model_type == "baseline":
        y_pred = predict_sklearn_model(
            model_path=model_spec.path,
            texts=texts,
            task=task,
            label_mapping=label_mapping,
        )

    elif model_spec.model_type == "phobert":
        y_pred = predict_phobert_model(
            model_path=model_spec.path,
            texts=texts,
            task=task,
            label_mapping=label_mapping,
            fallback_model_name=phobert_model_name,
            max_length=max_length,
            batch_size=phobert_batch_size,
            device=device,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_spec.model_type}")

    summary_metrics = compute_summary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
    )

    noise_type = str(eval_df["noise_type"].iloc[0])
    evaluation_scope = str(eval_df["evaluation_scope"].iloc[0])

    summary_row = {
        "task": task,
        "model_type": model_spec.model_type,
        "model_name": model_spec.model_name,
        "model_path": str(model_spec.path),
        "noise_type": noise_type,
        "evaluation_scope": evaluation_scope,
        "num_eval_samples": len(eval_df),
        **summary_metrics,
    }

    predictions_df = eval_df[[
        "id",
        "noise_type",
        "evaluation_scope",
        "text",
        "true_label",
        "is_changed",
        "changed_token_ratio",
        "severity",
        "original_text",
        "noisy_text",
    ]].copy()

    predictions_df["task"] = task
    predictions_df["model_type"] = model_spec.model_type
    predictions_df["model_name"] = model_spec.model_name
    predictions_df["pred_label"] = y_pred
    predictions_df["is_correct"] = predictions_df["true_label"] == predictions_df["pred_label"]

    report_rows = classification_report_rows(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        task=task,
        model_type=model_spec.model_type,
        model_name=model_spec.model_name,
        noise_type=noise_type,
        evaluation_scope=evaluation_scope,
    )

    return summary_row, predictions_df, report_rows


def compute_robustness_drop(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute absolute and relative macro-F1 drop against clean performance."""
    rows = []

    key_cols = ["task", "model_type", "model_name"]

    clean_df = results_df[
        (results_df["noise_type"] == "clean")
        & (results_df["evaluation_scope"] == "full")
    ].copy()

    clean_lookup = {
        tuple(row[col] for col in key_cols): row
        for _, row in clean_df.iterrows()
    }

    noisy_df = results_df[results_df["noise_type"] != "clean"].copy()

    for _, row in noisy_df.iterrows():
        key = tuple(row[col] for col in key_cols)
        clean_row = clean_lookup.get(key)

        if clean_row is None:
            continue

        clean_macro_f1 = float(clean_row["macro_f1"])
        noisy_macro_f1 = float(row["macro_f1"])
        absolute_drop = clean_macro_f1 - noisy_macro_f1
        relative_drop_pct = (
            absolute_drop / clean_macro_f1 * 100
            if clean_macro_f1 != 0
            else 0.0
        )

        rows.append({
            "task": row["task"],
            "model_type": row["model_type"],
            "model_name": row["model_name"],
            "noise_type": row["noise_type"],
            "evaluation_scope": row["evaluation_scope"],
            "clean_macro_f1": clean_macro_f1,
            "noisy_macro_f1": noisy_macro_f1,
            "absolute_macro_f1_drop": absolute_drop,
            "relative_macro_f1_drop_pct": relative_drop_pct,
            "clean_accuracy": float(clean_row["accuracy"]),
            "noisy_accuracy": float(row["accuracy"]),
            "absolute_accuracy_drop": float(clean_row["accuracy"]) - float(row["accuracy"]),
        })

    return pd.DataFrame(rows)


def rank_robustness(drop_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact robustness ranking table."""
    if drop_df.empty:
        return pd.DataFrame()

    full_df = drop_df[drop_df["evaluation_scope"] == "full"].copy()

    ranking = (
        full_df
        .groupby(["task", "model_type", "model_name"])
        .agg(
            mean_absolute_macro_f1_drop=("absolute_macro_f1_drop", "mean"),
            max_absolute_macro_f1_drop=("absolute_macro_f1_drop", "max"),
            mean_relative_macro_f1_drop_pct=("relative_macro_f1_drop_pct", "mean"),
        )
        .reset_index()
        .sort_values(["task", "mean_absolute_macro_f1_drop"], ascending=[True, True])
    )

    return ranking
