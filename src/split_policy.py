"""Split policy utilities.

All models must use the same train/dev/test split.
"""


def describe_split_policy() -> str:
    """Return the default split policy description."""
    return (
        "Use the public train/dev/test split if available. "
        "If no public split exists, create a stratified split with seed=42."
    )
