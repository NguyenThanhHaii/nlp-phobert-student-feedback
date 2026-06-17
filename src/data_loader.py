"""Utilities for loading and verifying dataset files.

Stage 1 will implement the real dataset loading logic.
"""

from pathlib import Path
from typing import Union

import pandas as pd


PathLike = Union[str, Path]


def read_csv_file(path: PathLike) -> pd.DataFrame:
    """Read a CSV file with a clear error message.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
