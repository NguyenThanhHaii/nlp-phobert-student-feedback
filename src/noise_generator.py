"""Controlled noise generation utilities.

Rules will be implemented after dataset verification.
"""

from dataclasses import dataclass


@dataclass
class ChangedSpan:
    """Metadata for one text transformation span.

    Token positions use the original whitespace-tokenized text.
    end_token is exclusive.
    """

    start_token: int
    end_token: int
    original: str
    replacement: str
    rule: str


def compute_changed_token_ratio(total_tokens: int, changed_spans: list[ChangedSpan]) -> float:
    """Compute changed token ratio from changed spans.

    changed_token_ratio =
    number of original tokens covered by changed spans / total original tokens
    """
    if total_tokens <= 0:
        return 0.0

    changed_count = sum(span.end_token - span.start_token for span in changed_spans)
    return changed_count / total_tokens
