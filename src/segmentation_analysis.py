"""Segmentation and tokenization analysis utilities."""


def count_compound_tokens(segmented_text: str) -> int:
    """Count Vietnamese compound tokens after word segmentation.

    A compound token is defined as a segmented token containing '_',
    for example: 'giảng_viên' or 'nhiệt_tình'.
    """
    if not segmented_text:
        return 0

    return sum(1 for token in segmented_text.split() if "_" in token)
