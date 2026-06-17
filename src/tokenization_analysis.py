"""Stage 5 utilities for segmentation and PhoBERT tokenization analysis.

This module compares clean and noisy Vietnamese student feedback before model
evaluation. It focuses on measurable text-level effects:

- whitespace/segmentation token count change
- PhoBERT subword count change
- subword inflation ratio
- tokenization change flag
- examples with large tokenization shifts

The default segmentation method is whitespace because it is reproducible and
requires no external Java/VnCoreNLP setup. Optional py_vncorenlp support is
included but not required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


def safe_json_loads(value):
    """Parse a JSON string safely."""
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        return json.loads(str(value))
    except Exception:
        return []


def whitespace_segment(text: str) -> list[str]:
    """Stable fallback tokenizer using whitespace splitting."""
    return str(text).split()


def normalize_token_list(tokens: Iterable[str]) -> list[str]:
    """Normalize tokens to strings and remove empty values."""
    return [str(token) for token in tokens if str(token).strip()]


@dataclass
class Segmenter:
    """Wrapper around a segmentation callable."""

    method: str
    segment_fn: Callable[[str], list[str]]

    def segment(self, text: str) -> list[str]:
        return normalize_token_list(self.segment_fn(str(text)))


def build_segmenter(method: str = "whitespace", py_vncorenlp_model_dir: str | None = None) -> Segmenter:
    """Build a Vietnamese segmenter.

    The project defaults to whitespace segmentation for reproducibility.
    Optional py_vncorenlp mode is available when the local environment is set up.
    """
    method = str(method).strip().lower()

    if method == "whitespace":
        return Segmenter(method="whitespace", segment_fn=whitespace_segment)

    if method == "py_vncorenlp":
        try:
            import py_vncorenlp

            model_dir = py_vncorenlp_model_dir or "data/external/vncorenlp"
            rdrsegmenter = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"],
                save_dir=model_dir,
            )

            def segment_with_vncorenlp(text: str) -> list[str]:
                segmented_sentences = rdrsegmenter.word_segment(str(text))
                joined = " ".join(segmented_sentences)
                return joined.split()

            return Segmenter(method="py_vncorenlp", segment_fn=segment_with_vncorenlp)

        except Exception as error:
            print("Could not initialize py_vncorenlp. Falling back to whitespace.")
            print("Reason:", error)
            return Segmenter(method="whitespace_fallback_from_py_vncorenlp", segment_fn=whitespace_segment)

    raise ValueError(f"Unsupported segmentation method: {method}")


@dataclass
class TokenizerWrapper:
    """Wrapper around PhoBERT tokenizer or a fallback tokenizer."""

    source: str
    tokenize_fn: Callable[[str], list[str]]

    def tokenize(self, text: str) -> list[str]:
        return normalize_token_list(self.tokenize_fn(str(text)))

    def count(self, text: str) -> int:
        return len(self.tokenize(text))


def build_phobert_tokenizer(model_name: str = "vinai/phobert-base", use_phobert_tokenizer: bool = True) -> TokenizerWrapper:
    """Build a PhoBERT tokenizer wrapper with fallback behavior."""
    if not use_phobert_tokenizer:
        return TokenizerWrapper(source="whitespace_fallback", tokenize_fn=whitespace_segment)

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        def tokenize(text: str) -> list[str]:
            return tokenizer.tokenize(str(text))

        return TokenizerWrapper(source=model_name, tokenize_fn=tokenize)

    except Exception as error:
        print("Could not load PhoBERT tokenizer. Falling back to whitespace.")
        print("Reason:", error)
        return TokenizerWrapper(source="whitespace_fallback_from_phobert", tokenize_fn=whitespace_segment)


def token_jaccard_distance(clean_tokens: list[str], noisy_tokens: list[str]) -> float:
    """Compute a simple set-based token distance."""
    clean_set = set(clean_tokens)
    noisy_set = set(noisy_tokens)

    if not clean_set and not noisy_set:
        return 0.0

    union = clean_set | noisy_set
    intersection = clean_set & noisy_set

    if not union:
        return 0.0

    return 1.0 - (len(intersection) / len(union))


def sequence_changed(clean_tokens: list[str], noisy_tokens: list[str]) -> bool:
    """Return whether the exact token sequence changed."""
    return clean_tokens != noisy_tokens


def ratio_with_zero_guard(numerator: float, denominator: float) -> float:
    """Compute numerator / denominator while avoiding division by zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def analyze_clean_noisy_pair(
    row: pd.Series,
    segmenter: Segmenter,
    tokenizer: TokenizerWrapper,
) -> dict:
    """Analyze one clean/noisy row pair."""
    clean_text = str(row["original_text"])
    noisy_text = str(row["noisy_text"])

    clean_segments = segmenter.segment(clean_text)
    noisy_segments = segmenter.segment(noisy_text)

    clean_subwords = tokenizer.tokenize(clean_text)
    noisy_subwords = tokenizer.tokenize(noisy_text)

    num_segments_clean = len(clean_segments)
    num_segments_noisy = len(noisy_segments)
    num_subwords_clean = len(clean_subwords)
    num_subwords_noisy = len(noisy_subwords)

    subword_delta = num_subwords_noisy - num_subwords_clean
    segment_delta = num_segments_noisy - num_segments_clean

    return {
        "id": row["id"],
        "noise_type": row["noise_type"],
        "original_text": clean_text,
        "noisy_text": noisy_text,
        "sentiment_label": row["sentiment_label"],
        "topic_label": row["topic_label"],
        "changed_token_ratio": row.get("changed_token_ratio", 0.0),
        "severity": row.get("severity", ""),
        "is_changed": bool(row.get("is_changed", clean_text != noisy_text)),
        "num_segments_clean": num_segments_clean,
        "num_segments_noisy": num_segments_noisy,
        "segment_delta": segment_delta,
        "segment_count_ratio": ratio_with_zero_guard(num_segments_noisy, num_segments_clean),
        "segmentation_change_flag": sequence_changed(clean_segments, noisy_segments),
        "segmentation_jaccard_distance": token_jaccard_distance(clean_segments, noisy_segments),
        "num_subwords_clean": num_subwords_clean,
        "num_subwords_noisy": num_subwords_noisy,
        "subword_delta": subword_delta,
        "subword_inflation_ratio": ratio_with_zero_guard(num_subwords_noisy, num_subwords_clean),
        "subword_change_flag": sequence_changed(clean_subwords, noisy_subwords),
        "subword_jaccard_distance": token_jaccard_distance(clean_subwords, noisy_subwords),
        "clean_segments": json.dumps(clean_segments, ensure_ascii=False),
        "noisy_segments": json.dumps(noisy_segments, ensure_ascii=False),
        "clean_subwords": json.dumps(clean_subwords, ensure_ascii=False),
        "noisy_subwords": json.dumps(noisy_subwords, ensure_ascii=False),
        "changed_spans": row.get("changed_spans", "[]"),
    }


def analyze_noisy_dataframe(
    noisy_df: pd.DataFrame,
    segmenter: Segmenter,
    tokenizer: TokenizerWrapper,
) -> pd.DataFrame:
    """Analyze all rows in one noisy dataframe."""
    required_cols = [
        "id",
        "original_text",
        "noisy_text",
        "sentiment_label",
        "topic_label",
        "noise_type",
    ]

    missing = [col for col in required_cols if col not in noisy_df.columns]
    if missing:
        raise KeyError(f"Missing required noisy columns: {missing}")

    rows = [
        analyze_clean_noisy_pair(row=row, segmenter=segmenter, tokenizer=tokenizer)
        for _, row in noisy_df.iterrows()
    ]

    return pd.DataFrame(rows)


def summarize_analysis(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Create per-noise-type summary for segmentation and subword effects."""
    grouped = analysis_df.groupby("noise_type")

    summary = grouped.agg(
        num_rows=("id", "count"),
        num_changed_rows=("is_changed", "sum"),
        mean_changed_token_ratio=("changed_token_ratio", "mean"),
        mean_segment_delta=("segment_delta", "mean"),
        median_segment_delta=("segment_delta", "median"),
        segmentation_change_rate=("segmentation_change_flag", "mean"),
        mean_segmentation_jaccard_distance=("segmentation_jaccard_distance", "mean"),
        mean_subword_delta=("subword_delta", "mean"),
        median_subword_delta=("subword_delta", "median"),
        mean_subword_inflation_ratio=("subword_inflation_ratio", "mean"),
        median_subword_inflation_ratio=("subword_inflation_ratio", "median"),
        subword_change_rate=("subword_change_flag", "mean"),
        mean_subword_jaccard_distance=("subword_jaccard_distance", "mean"),
    ).reset_index()

    for col in ["segmentation_change_rate", "subword_change_rate"]:
        summary[col] = summary[col] * 100

    return summary


def summarize_changed_only(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Create summary using only rows that were actually changed."""
    changed_df = analysis_df[analysis_df["is_changed"]].copy()

    if changed_df.empty:
        return pd.DataFrame()

    summary = summarize_analysis(changed_df)
    summary = summary.rename(columns={
        "num_rows": "num_changed_subset_rows",
    })

    return summary


def select_high_inflation_examples(
    analysis_df: pd.DataFrame,
    top_n: int = 15,
) -> pd.DataFrame:
    """Select examples with strongest subword inflation per noise type."""
    example_frames = []

    for noise_type, group_df in analysis_df.groupby("noise_type"):
        ranked = (
            group_df[group_df["is_changed"]]
            .sort_values(
                ["subword_delta", "subword_inflation_ratio", "changed_token_ratio"],
                ascending=[False, False, False],
            )
            .head(top_n)
        )

        if ranked.empty:
            ranked = group_df.head(top_n)

        example_frames.append(ranked)

    if not example_frames:
        return pd.DataFrame()

    columns = [
        "id",
        "noise_type",
        "original_text",
        "noisy_text",
        "sentiment_label",
        "topic_label",
        "changed_token_ratio",
        "severity",
        "num_segments_clean",
        "num_segments_noisy",
        "segment_delta",
        "num_subwords_clean",
        "num_subwords_noisy",
        "subword_delta",
        "subword_inflation_ratio",
        "clean_segments",
        "noisy_segments",
        "clean_subwords",
        "noisy_subwords",
        "changed_spans",
    ]

    result = pd.concat(example_frames, ignore_index=True)
    return result[[col for col in columns if col in result.columns]]


def load_noisy_files(noisy_dir: Path, pattern: str = "04_test_*.csv") -> dict[str, pd.DataFrame]:
    """Load all Stage 4 noisy files."""
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
