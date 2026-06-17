"""Utilities for Stage 3 PhoBERT fine-tuning."""

from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Any

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch may not be installed in local analysis-only environments.
        pass


def resolve_project_path(project_root: Path, maybe_relative_path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    maybe_relative_path = Path(maybe_relative_path)
    if maybe_relative_path.is_absolute():
        return maybe_relative_path
    return project_root / maybe_relative_path


def load_label_info(label_mapping_path: str | Path, task: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Load ordered labels and mapping dictionaries for a task."""
    with Path(label_mapping_path).open("r", encoding="utf-8") as f:
        label_mapping = json.load(f)

    if task not in label_mapping:
        raise KeyError(f"Task not found in label mapping: {task}")

    id_to_name_raw = label_mapping[task]["id_to_name"]

    ordered_items = sorted(id_to_name_raw.items(), key=lambda item: int(item[0]))
    labels = [str(name) for _, name in ordered_items]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}

    return labels, label2id, id2label


def basic_text_clean(text: object) -> str:
    """Minimal text cleanup for PhoBERT input."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def maybe_word_segment(text: object, preprocessing_config: dict[str, Any]) -> str:
    """Optionally apply Vietnamese word segmentation.

    Default Stage 3 configuration keeps this disabled for environment stability.
    If enabled with method="underthesea", install underthesea first and accept that
    its segmenter is not identical to the RDRSegmenter used in PhoBERT pretraining.
    """
    cleaned = basic_text_clean(text)

    enabled = bool(preprocessing_config.get("word_segmentation_enabled", False))
    method = str(preprocessing_config.get("word_segmentation_method", "none")).lower()

    if not enabled or method == "none":
        return cleaned

    if method == "underthesea":
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise ImportError(
                "word_segmentation_method='underthesea' requires package underthesea. "
                "Install it or disable word segmentation in the config."
            ) from exc

        return word_tokenize(cleaned, format="text")

    raise ValueError(f"Unsupported word segmentation method: {method}")


def prepare_texts(texts: pd.Series, preprocessing_config: dict[str, Any]) -> list[str]:
    """Prepare a pandas text series for PhoBERT tokenization."""
    return [maybe_word_segment(text, preprocessing_config) for text in texts.tolist()]


def read_processed_splits(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read processed train/dev/test CSV files from Stage 1."""
    data_dir = Path(data_dir)

    paths = {
        "train": data_dir / "train.csv",
        "dev": data_dir / "dev.csv",
        "test": data_dir / "test.csv",
    }

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing processed {split} split: {path}")

    return {split: pd.read_csv(path) for split, path in paths.items()}


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert a small DataFrame to a markdown table without requiring tabulate."""
    if df.empty:
        return "_Empty table_"

    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].astype(str)

    headers = safe_df.columns.tolist()
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in safe_df.iterrows():
        lines.append("| " + " | ".join(row[col] for col in headers) + " |")

    return "\n".join(lines)
