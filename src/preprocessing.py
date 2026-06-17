"""Text preprocessing utilities for baseline experiments.

This module intentionally keeps preprocessing minimal for clean baseline models.
Do not remove accents, normalize abbreviations, or correct spelling here. Those
steps are part of later noisy-data and optional normalization experiments.
"""

from __future__ import annotations

import pandas as pd


def basic_text_clean(text: object) -> str:
    """Convert input to string and strip surrounding whitespace."""
    if pd.isna(text):
        return ""
    return str(text).strip()


def prepare_text_series(series: pd.Series) -> pd.Series:
    """Prepare a pandas Series of text for baseline vectorizers."""
    return series.apply(basic_text_clean)
