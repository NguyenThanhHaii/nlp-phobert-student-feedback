from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make src importable when running from project root.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_LABEL_MAPPING = {
    "sentiment": {
        "0": "negative",
        "1": "neutral",
        "2": "positive",
    },
    "topic": {
        "0": "lecturer",
        "1": "training_program",
        "2": "facility",
        "3": "others",
    },
}

DISPLAY_LABELS = {
    "negative": "Tiêu cực",
    "neutral": "Trung tính",
    "positive": "Tích cực",
    "lecturer": "Giảng viên",
    "training_program": "Chương trình đào tạo",
    "facility": "Cơ sở vật chất",
    "others": "Khác",
}

TASK_LABELS = {
    "sentiment": ["negative", "neutral", "positive"],
    "topic": ["lecturer", "training_program", "facility", "others"],
}

BEST_BASELINE_MODEL = {
    # Stage 2 best by dev Macro-F1.
    "sentiment": "tfidf_char_svm",
    "topic": "tfidf_word_svm",
}


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_label_mapping() -> dict[str, dict[str, str]]:
    mapping_path = project_path("configs", "label_mapping.json")

    if not mapping_path.exists():
        return DEFAULT_LABEL_MAPPING

    with open(mapping_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        task: {str(k): str(v).lower() for k, v in raw[task]["id_to_name"].items()}
        for task in ["sentiment", "topic"]
    }


LABEL_MAPPING = load_label_mapping()


def label_names(task: str) -> list[str]:
    mapping = LABEL_MAPPING.get(task, DEFAULT_LABEL_MAPPING[task])
    return [
        mapping[key]
        for key in sorted(mapping.keys(), key=lambda value: int(value))
    ]


def normalize_prediction_label(task: str, pred: Any) -> str:
    mapping = LABEL_MAPPING.get(task, DEFAULT_LABEL_MAPPING[task])

    if isinstance(pred, (int, np.integer)):
        return mapping.get(str(int(pred)), str(pred).lower())

    pred_str = str(pred)
    if pred_str in mapping:
        return mapping[pred_str]

    return pred_str.lower()


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.max(values)
    exp_values = np.exp(values)
    denom = exp_values.sum()
    if denom == 0:
        return np.ones_like(exp_values) / len(exp_values)
    return exp_values / denom


def simple_rule_normalize(text: str) -> str:
    """Small runtime normalizer for demo input.

    This mirrors the purpose of Stage 8 without requiring the full notebook.
    """
    current = str(text).strip()

    # Reduce elongated characters: hayyy -> hay, tốttt -> tốt.
    current = re.sub(r"(.)\1{2,}", r"\1", current, flags=re.IGNORECASE)

    replacements = {
        # Domain abbreviations
        "gv": "giảng viên",
        "sv": "sinh viên",
        "hs": "học sinh",
        "hp": "học phần",
        "mh": "môn học",
        "bt": "bài tập",
        "bkt": "bài kiểm tra",
        "kt": "kiểm tra",
        "csvc": "cơ sở vật chất",
        "ctdt": "chương trình đào tạo",
        # Teencode
        "ko": "không",
        "k": "không",
        "k0": "không",
        "khong": "không",
        "dc": "được",
        "đc": "được",
        "duoc": "được",
        "lun": "luôn",
        "nhìu": "nhiều",
        "nhieu": "nhiều",
        "vs": "với",
        "bít": "biết",
        "bit": "biết",
        # Common no-accent phrases / words
        "giang vien": "giảng viên",
        "sinh vien": "sinh viên",
        "hoc phan": "học phần",
        "mon hoc": "môn học",
        "co so vat chat": "cơ sở vật chất",
        "phong hoc": "phòng học",
        "chuong trinh dao tao": "chương trình đào tạo",
        "nhiet tinh": "nhiệt tình",
        "de hieu": "dễ hiểu",
        "kho hieu": "khó hiểu",
        "ro rang": "rõ ràng",
        "rat tot": "rất tốt",
        "khong tot": "không tốt",
        "tot": "tốt",
        "kem": "kém",
    }

    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", flags=re.IGNORECASE)
        current = pattern.sub(target, current)

    return current


def baseline_model_path(task: str) -> Path | None:
    preferred_name = BEST_BASELINE_MODEL[task]

    candidates = [
        project_path("models", "baselines", task, f"{preferred_name}.joblib"),
        project_path("models", task, f"{preferred_name}.joblib"),
    ]

    for path in candidates:
        if path.exists():
            return path

    # Fallback to any non-majority model.
    fallback_dirs = [
        project_path("models", "baselines", task),
        project_path("models", task),
    ]

    for model_dir in fallback_dirs:
        if not model_dir.exists():
            continue

        for path in sorted(model_dir.glob("*.joblib")):
            if path.stem != "majority_class":
                return path

    return None


def phobert_model_path(task: str) -> Path:
    return project_path("models", "phobert", task, "best_model")


@lru_cache(maxsize=4)
def load_baseline_model(task: str):
    path = baseline_model_path(task)
    if path is None:
        raise FileNotFoundError(f"No baseline model found for task={task}")

    return joblib.load(path), path


@lru_cache(maxsize=4)
def load_phobert_model(task: str):
    model_dir = phobert_model_path(task)

    if not model_dir.exists():
        raise FileNotFoundError(f"No PhoBERT model found for task={task}: {model_dir}")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "PhoBERT requires torch and transformers. Install them if you want to use the PhoBERT demo mode."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device, model_dir


def predict_baseline(task: str, text: str) -> dict[str, Any]:
    model, path = load_baseline_model(task)
    labels = label_names(task)

    pred = model.predict([text])[0]
    label = normalize_prediction_label(task, pred)

    scores = None
    confidence = None

    if hasattr(model, "decision_function"):
        try:
            raw_scores = model.decision_function([text])
            raw_scores = np.asarray(raw_scores)

            if raw_scores.ndim == 1:
                # Binary case, not expected here but handled safely.
                raw_scores = np.stack([-raw_scores, raw_scores], axis=1)

            probs = stable_softmax(raw_scores[0])

            # Some sklearn pipelines preserve classes_ on final estimator.
            classes = getattr(model, "classes_", None)
            if classes is None and hasattr(model, "named_steps"):
                final_step = list(model.named_steps.values())[-1]
                classes = getattr(final_step, "classes_", None)

            if classes is not None:
                mapped_labels = [normalize_prediction_label(task, item) for item in classes]
            else:
                mapped_labels = labels

            scores = {
                mapped_labels[i]: float(probs[i])
                for i in range(min(len(mapped_labels), len(probs)))
            }
            confidence = float(max(scores.values())) if scores else None
        except Exception:
            scores = None
            confidence = None

    return {
        "task": task,
        "model_type": "baseline",
        "model_name": path.stem,
        "label": label,
        "confidence": confidence,
        "scores": scores,
    }


def get_phobert_id2label(task: str, model) -> dict[int, str]:
    config_mapping = getattr(model.config, "id2label", None)

    if isinstance(config_mapping, dict):
        normalized = {}
        valid = True

        for key, value in config_mapping.items():
            try:
                key_int = int(key)
            except Exception:
                valid = False
                break

            value_str = str(value).lower()
            if value_str.startswith("label_"):
                valid = False
                break

            normalized[key_int] = value_str

        if valid and normalized:
            return normalized

    return {int(k): v for k, v in LABEL_MAPPING.get(task, DEFAULT_LABEL_MAPPING[task]).items()}


def predict_phobert(task: str, text: str) -> dict[str, Any]:
    import torch

    tokenizer, model, device, model_dir = load_phobert_model(task)
    id2label = get_phobert_id2label(task, model)

    encoded = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu().numpy()

    pred_id = int(np.argmax(probs))
    label = id2label[pred_id]
    scores = {
        id2label[i]: float(probs[i])
        for i in range(len(probs))
        if i in id2label
    }

    return {
        "task": task,
        "model_type": "phobert",
        "model_name": "phobert_base",
        "label": label,
        "confidence": float(probs[pred_id]),
        "scores": scores,
    }


def available_models() -> dict[str, Any]:
    info = {
        "project_root": str(PROJECT_ROOT),
        "tasks": {},
    }

    for task in ["sentiment", "topic"]:
        base_path = baseline_model_path(task)
        pho_path = phobert_model_path(task)

        info["tasks"][task] = {
            "labels": label_names(task),
            "baseline": {
                "available": base_path is not None,
                "path": str(base_path) if base_path else None,
                "default_model": BEST_BASELINE_MODEL[task],
            },
            "phobert": {
                "available": pho_path.exists(),
                "path": str(pho_path),
                "default_model": "phobert_base",
            },
        }

    return info


def choose_model(task: str, requested_model: str) -> str:
    if requested_model == "baseline":
        return "baseline"
    if requested_model == "phobert":
        return "phobert"

    # auto: prefer PhoBERT if checkpoint exists; otherwise baseline.
    if phobert_model_path(task).exists():
        return "phobert"
    return "baseline"


def predict(task: str, text: str, model_choice: str) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    selected = choose_model(task, model_choice)

    try:
        if selected == "phobert":
            return predict_phobert(task, text), warnings
        return predict_baseline(task, text), warnings

    except Exception as exc:
        if selected == "phobert" and model_choice == "auto":
            warnings.append(
                f"PhoBERT unavailable for task={task}; falling back to baseline. Detail: {exc}"
            )
            return predict_baseline(task, text), warnings
        raise


def label_display(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)
