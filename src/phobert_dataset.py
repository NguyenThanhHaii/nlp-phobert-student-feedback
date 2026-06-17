"""Torch dataset for PhoBERT text classification."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset


class PhoBERTClassificationDataset(Dataset):
    """Dataset wrapper for tokenized PhoBERT classification inputs."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        tokenizer,
        label2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.labels = [label2id[str(label)] for label in labels]

        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
