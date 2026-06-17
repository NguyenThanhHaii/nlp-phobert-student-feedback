"""Controlled Vietnamese noise generation for Stage 4.

The design goal is reproducibility and traceability, not aggressive corruption.
Every generated row keeps the original labels and records the transformation
metadata needed for later robustness and tokenization analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
import unicodedata
from typing import Callable, Iterable

import pandas as pd


@dataclass
class TokenItem:
    """A token plus the original whitespace-token indices it represents."""

    token: str
    orig_indices: list[int]


def remove_vietnamese_accents(text: str) -> str:
    """Remove Vietnamese diacritics from text while keeping base characters."""
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", without_marks)


def split_token_affixes(token: str) -> tuple[str, str, str]:
    """Split a whitespace token into prefix, alnum core, and suffix.

    This keeps punctuation such as commas and periods attached after replacement.
    """
    if not token:
        return "", "", ""

    start = 0
    while start < len(token) and not token[start].isalnum():
        start += 1

    end = len(token)
    while end > start and not token[end - 1].isalnum():
        end -= 1

    return token[:start], token[start:end], token[end:]


def normalize_token_for_match(token: str) -> str:
    """Normalize a token for rule matching."""
    _, core, _ = split_token_affixes(token)
    return core.lower()


def make_items(text: str) -> list[TokenItem]:
    """Create token items from whitespace tokens."""
    tokens = str(text).split()
    return [
        TokenItem(token=token, orig_indices=[idx])
        for idx, token in enumerate(tokens)
    ]


def items_to_text(items: Iterable[TokenItem]) -> str:
    """Convert token items back to text."""
    return " ".join(item.token for item in items)


def span_from_items(
    items: list[TokenItem],
    start: int,
    end: int,
    replacement: str,
    rule: str,
) -> dict:
    """Create a changed span using original token indices."""
    affected = []
    for item in items[start:end]:
        affected.extend(item.orig_indices)

    original = " ".join(item.token for item in items[start:end])

    return {
        "start_token": min(affected),
        "end_token": max(affected) + 1,
        "original": original,
        "replacement": replacement,
        "rule": rule,
    }


def make_replacement_token(original_items: list[TokenItem], replacement: str) -> str:
    """Attach original prefix/suffix punctuation around a replacement."""
    first_prefix, _, _ = split_token_affixes(original_items[0].token)
    _, _, last_suffix = split_token_affixes(original_items[-1].token)
    return f"{first_prefix}{replacement}{last_suffix}"


def apply_phrase_replacements(
    items: list[TokenItem],
    replacements: dict[str, str | list[str]],
    rule: str,
    rng: random.Random,
    max_replacements: int = 2,
) -> tuple[list[TokenItem], list[dict]]:
    """Apply phrase-level replacements to token items."""
    if not replacements:
        return items, []

    phrase_entries = []
    for phrase, replacement in replacements.items():
        phrase_tokens = str(phrase).lower().split()
        phrase_entries.append((phrase_tokens, replacement))

    # Longest phrase first to avoid replacing a subphrase prematurely.
    phrase_entries.sort(key=lambda item: len(item[0]), reverse=True)

    new_items: list[TokenItem] = []
    spans: list[dict] = []

    idx = 0
    replacement_count = 0

    while idx < len(items):
        matched = False

        if replacement_count < max_replacements:
            for phrase_tokens, replacement_value in phrase_entries:
                phrase_len = len(phrase_tokens)
                window = items[idx: idx + phrase_len]

                if len(window) != phrase_len:
                    continue

                window_tokens = [
                    normalize_token_for_match(item.token)
                    for item in window
                ]

                if window_tokens != phrase_tokens:
                    continue

                if isinstance(replacement_value, list):
                    replacement = rng.choice(replacement_value)
                else:
                    replacement = str(replacement_value)

                replacement_token = make_replacement_token(window, replacement)
                orig_indices = []
                for item in window:
                    orig_indices.extend(item.orig_indices)

                spans.append(
                    span_from_items(
                        items=items,
                        start=idx,
                        end=idx + phrase_len,
                        replacement=replacement_token,
                        rule=rule,
                    )
                )

                new_items.append(
                    TokenItem(
                        token=replacement_token,
                        orig_indices=orig_indices,
                    )
                )

                idx += phrase_len
                replacement_count += 1
                matched = True
                break

        if not matched:
            new_items.append(items[idx])
            idx += 1

    return new_items, spans


def apply_no_accent_to_items(items: list[TokenItem]) -> tuple[list[TokenItem], list[dict]]:
    """Apply accent removal token by token."""
    new_items: list[TokenItem] = []
    spans: list[dict] = []

    for idx, item in enumerate(items):
        replacement = remove_vietnamese_accents(item.token)

        if replacement != item.token:
            spans.append(
                span_from_items(
                    items=items,
                    start=idx,
                    end=idx + 1,
                    replacement=replacement,
                    rule="no_accent",
                )
            )
            new_items.append(TokenItem(token=replacement, orig_indices=item.orig_indices))
        else:
            new_items.append(item)

    return new_items, spans


def token_has_diacritic(token: str) -> bool:
    """Return True if a token contains at least one diacritic."""
    _, core, _ = split_token_affixes(token)
    return remove_vietnamese_accents(core) != core


def remove_one_diacritic_from_token(token: str) -> str:
    """Remove diacritics from one character in a token core."""
    prefix, core, suffix = split_token_affixes(token)

    for char in core:
        no_accent = remove_vietnamese_accents(char)
        if no_accent != char:
            return prefix + core.replace(char, no_accent, 1) + suffix

    return token


def swap_adjacent_chars(token: str) -> str:
    """Create a small adjacent-character swap typo in the token core."""
    prefix, core, suffix = split_token_affixes(token)

    if len(core) < 4:
        return token

    chars = list(core)
    swap_idx = min(2, len(chars) - 2)
    chars[swap_idx], chars[swap_idx + 1] = chars[swap_idx + 1], chars[swap_idx]

    return prefix + "".join(chars) + suffix


def apply_typo_to_items(
    items: list[TokenItem],
    rng: random.Random,
    max_typos: int = 1,
    min_token_length: int = 4,
    prefer_diacritic_typo: bool = True,
) -> tuple[list[TokenItem], list[dict]]:
    """Apply light typo noise to up to max_typos tokens."""
    new_items = [TokenItem(token=item.token, orig_indices=item.orig_indices) for item in items]
    spans: list[dict] = []

    candidate_indices = []
    for idx, item in enumerate(items):
        _, core, _ = split_token_affixes(item.token)
        if len(core) >= min_token_length and not core.isnumeric():
            candidate_indices.append(idx)

    if not candidate_indices:
        return new_items, spans

    if prefer_diacritic_typo:
        diacritic_candidates = [
            idx for idx in candidate_indices
            if token_has_diacritic(items[idx].token)
        ]
        if diacritic_candidates:
            candidate_indices = diacritic_candidates

    rng.shuffle(candidate_indices)

    changed = 0
    for idx in candidate_indices:
        original_token = new_items[idx].token

        if token_has_diacritic(original_token):
            replacement = remove_one_diacritic_from_token(original_token)
        else:
            replacement = swap_adjacent_chars(original_token)

        if replacement == original_token:
            continue

        spans.append(
            span_from_items(
                items=items,
                start=idx,
                end=idx + 1,
                replacement=replacement,
                rule="typo",
            )
        )

        new_items[idx] = TokenItem(
            token=replacement,
            orig_indices=new_items[idx].orig_indices,
        )

        changed += 1
        if changed >= max_typos:
            break

    return new_items, spans


def elongate_token(token: str, repeat_count: int = 2) -> str:
    """Repeat the last core character of a token."""
    prefix, core, suffix = split_token_affixes(token)

    if not core:
        return token

    return prefix + core + (core[-1] * repeat_count) + suffix


def apply_elongation_to_items(
    items: list[TokenItem],
    rng: random.Random,
    max_elongations: int = 1,
    min_token_length: int = 3,
    repeat_count: int = 2,
) -> tuple[list[TokenItem], list[dict]]:
    """Apply character elongation to up to max_elongations tokens."""
    new_items = [TokenItem(token=item.token, orig_indices=item.orig_indices) for item in items]
    spans: list[dict] = []

    candidate_indices = []
    for idx, item in enumerate(items):
        _, core, _ = split_token_affixes(item.token)
        if len(core) >= min_token_length and not core.isnumeric():
            candidate_indices.append(idx)

    if not candidate_indices:
        return new_items, spans

    rng.shuffle(candidate_indices)

    changed = 0
    for idx in candidate_indices:
        original_token = new_items[idx].token
        replacement = elongate_token(original_token, repeat_count=repeat_count)

        if replacement == original_token:
            continue

        spans.append(
            span_from_items(
                items=items,
                start=idx,
                end=idx + 1,
                replacement=replacement,
                rule="elongation",
            )
        )

        new_items[idx] = TokenItem(
            token=replacement,
            orig_indices=new_items[idx].orig_indices,
        )

        changed += 1
        if changed >= max_elongations:
            break

    return new_items, spans


def changed_token_ratio(original_text: str, changed_spans: list[dict]) -> float:
    """Compute changed-token ratio from original token spans."""
    total_tokens = len(str(original_text).split())

    if total_tokens == 0:
        return 0.0

    changed_indices = set()
    for span in changed_spans:
        for idx in range(int(span["start_token"]), int(span["end_token"])):
            changed_indices.add(idx)

    return len(changed_indices) / total_tokens


def severity_from_ratio(
    ratio: float,
    low_max_ratio: float = 0.20,
    medium_max_ratio: float = 0.40,
) -> str:
    """Map changed-token ratio to severity."""
    if ratio == 0:
        return "none"
    if ratio <= low_max_ratio:
        return "low"
    if ratio <= medium_max_ratio:
        return "medium"
    return "high"


def apply_noise(
    text: str,
    noise_type: str,
    config: dict,
    rng: random.Random,
) -> tuple[str, list[dict]]:
    """Apply one configured noise type to text."""
    items = make_items(text)

    if noise_type == "no_accent":
        new_items, spans = apply_no_accent_to_items(items)
        return items_to_text(new_items), spans

    if noise_type == "domain_abbreviation":
        domain_config = config["domain_abbreviation"]
        new_items, spans = apply_phrase_replacements(
            items=items,
            replacements=domain_config["replacements"],
            rule="domain_abbreviation",
            rng=rng,
            max_replacements=int(domain_config.get("max_replacements_per_sentence", 2)),
        )
        return items_to_text(new_items), spans

    if noise_type == "teencode_colloquial":
        teencode_config = config["teencode_colloquial"]
        new_items, spans = apply_phrase_replacements(
            items=items,
            replacements=teencode_config["replacements"],
            rule="teencode_colloquial",
            rng=rng,
            max_replacements=int(teencode_config.get("max_replacements_per_sentence", 2)),
        )
        return items_to_text(new_items), spans

    if noise_type == "typo":
        typo_config = config["typo"]
        new_items, spans = apply_typo_to_items(
            items=items,
            rng=rng,
            max_typos=int(typo_config.get("max_typos_per_sentence", 1)),
            min_token_length=int(typo_config.get("min_token_length", 4)),
            prefer_diacritic_typo=bool(typo_config.get("prefer_diacritic_typo", True)),
        )
        return items_to_text(new_items), spans

    if noise_type == "elongation":
        elongation_config = config["elongation"]
        new_items, spans = apply_elongation_to_items(
            items=items,
            rng=rng,
            max_elongations=int(elongation_config.get("max_elongations_per_sentence", 1)),
            min_token_length=int(elongation_config.get("min_token_length", 3)),
            repeat_count=int(elongation_config.get("repeat_count", 2)),
        )
        return items_to_text(new_items), spans

    if noise_type == "mixed_noise":
        return apply_mixed_noise(text=text, config=config, rng=rng)

    raise ValueError(f"Unsupported noise type: {noise_type}")


def apply_mixed_noise(
    text: str,
    config: dict,
    rng: random.Random,
) -> tuple[str, list[dict]]:
    """Apply mixed noise according to a fixed protocol."""
    mixed_config = config["mixed_noise"]
    apply_order = mixed_config.get("apply_order", [])

    items = make_items(text)
    all_spans: list[dict] = []

    applied_count = 0
    max_noise_types = int(mixed_config.get("max_noise_types_per_sentence", 3))

    for step in apply_order:
        if applied_count >= max_noise_types:
            break

        before_text = items_to_text(items)

        if step == "domain_abbreviation":
            domain_config = config["domain_abbreviation"]
            items, spans = apply_phrase_replacements(
                items=items,
                replacements=domain_config["replacements"],
                rule="domain_abbreviation",
                rng=rng,
                max_replacements=1,
            )

        elif step == "teencode_colloquial":
            teencode_config = config["teencode_colloquial"]
            items, spans = apply_phrase_replacements(
                items=items,
                replacements=teencode_config["replacements"],
                rule="teencode_colloquial",
                rng=rng,
                max_replacements=1,
            )

        elif step == "char_noise":
            char_choice = mixed_config.get("char_noise_choice", {"typo": 0.5, "elongation": 0.5})
            typo_weight = float(char_choice.get("typo", 0.5))
            elongation_weight = float(char_choice.get("elongation", 0.5))
            selected = rng.choices(
                ["typo", "elongation"],
                weights=[typo_weight, elongation_weight],
                k=1,
            )[0]

            if selected == "typo":
                typo_config = config["typo"]
                items, spans = apply_typo_to_items(
                    items=items,
                    rng=rng,
                    max_typos=1,
                    min_token_length=int(typo_config.get("min_token_length", 4)),
                    prefer_diacritic_typo=bool(typo_config.get("prefer_diacritic_typo", True)),
                )
            else:
                elongation_config = config["elongation"]
                items, spans = apply_elongation_to_items(
                    items=items,
                    rng=rng,
                    max_elongations=1,
                    min_token_length=int(elongation_config.get("min_token_length", 3)),
                    repeat_count=int(elongation_config.get("repeat_count", 2)),
                )

        elif step == "no_accent":
            if not bool(mixed_config.get("mixed_no_accent", True)):
                spans = []
            else:
                items, spans = apply_no_accent_to_items(items)

        else:
            spans = []

        after_text = items_to_text(items)

        if spans and after_text != before_text:
            all_spans.extend(spans)
            applied_count += 1

    return items_to_text(items), all_spans


def default_subword_counter(text: str) -> int:
    """Fallback subword counter based on whitespace tokenization."""
    return len(str(text).split())


def generate_noisy_dataset(
    df: pd.DataFrame,
    noise_type: str,
    config: dict,
    subword_counter: Callable[[str], int] | None = None,
) -> pd.DataFrame:
    """Generate one noisy dataset from a clean test DataFrame."""
    if subword_counter is None:
        subword_counter = default_subword_counter

    seed = int(config.get("seed", 42))
    rng = random.Random(seed + sum(ord(char) for char in noise_type))

    required_columns = ["id", "text", "sentiment_label", "topic_label"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns for noise generation: {missing}")

    rows = []

    low_max = float(config.get("severity", {}).get("low_max_ratio", 0.20))
    medium_max = float(config.get("severity", {}).get("medium_max_ratio", 0.40))
    max_changed_ratio = float(
        config.get("mixed_noise", {}).get("max_changed_token_ratio", 0.40)
    )

    for _, row in df.iterrows():
        original_text = str(row["text"])
        noisy_text, spans = apply_noise(
            text=original_text,
            noise_type=noise_type,
            config=config,
            rng=rng,
        )

        ratio = changed_token_ratio(original_text, spans)
        severity = severity_from_ratio(
            ratio=ratio,
            low_max_ratio=low_max,
            medium_max_ratio=medium_max,
        )

        rows.append({
            "id": row["id"],
            "original_text": original_text,
            "noisy_text": noisy_text,
            "sentiment_label": row["sentiment_label"],
            "topic_label": row["topic_label"],
            "noise_type": noise_type,
            "severity": severity,
            "changed_token_ratio": ratio,
            "changed_spans": json.dumps(spans, ensure_ascii=False),
            "num_words_clean": len(original_text.split()),
            "num_words_noisy": len(noisy_text.split()),
            "num_subwords_clean": subword_counter(original_text),
            "num_subwords_noisy": subword_counter(noisy_text),
            "is_changed": noisy_text != original_text,
            "exceeds_max_changed_token_ratio": (
                ratio > max_changed_ratio if noise_type == "mixed_noise" else False
            ),
        })

    return pd.DataFrame(rows)


def summarize_noisy_dataset(noisy_df: pd.DataFrame) -> dict:
    """Summarize one noisy dataset."""
    total = len(noisy_df)
    changed = int(noisy_df["is_changed"].sum())

    return {
        "noise_type": noisy_df["noise_type"].iloc[0] if total else "",
        "num_rows": total,
        "num_changed_rows": changed,
        "changed_row_percentage": round(changed / total * 100, 2) if total else 0,
        "mean_changed_token_ratio": noisy_df["changed_token_ratio"].mean(),
        "median_changed_token_ratio": noisy_df["changed_token_ratio"].median(),
        "max_changed_token_ratio": noisy_df["changed_token_ratio"].max(),
        "mean_subword_delta": (
            noisy_df["num_subwords_noisy"] - noisy_df["num_subwords_clean"]
        ).mean(),
    }
