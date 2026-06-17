"""Basic text preprocessing utilities.

Keep preprocessing conservative. Do not change labels or meaning.
"""


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace in a text string."""
    if text is None:
        return ""
    return " ".join(str(text).split())
