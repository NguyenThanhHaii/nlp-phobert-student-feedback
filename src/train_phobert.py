"""Stage 3 script: fine-tune PhoBERT on clean UIT-VSFC data.

Example:
    python src/train_phobert.py --task sentiment --config configs/03_phobert_training_config.yaml
    python src/train_phobert.py --task topic --config configs/03_phobert_training_config.yaml
    python src/train_phobert.py --task all --config configs/03_phobert_training_config.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phobert_dataset import PhoBERTClassificationDataset
from src.phobert_utils import (
    dataframe_to_markdown,
    load_label_info,
    load_yaml,
    prepare_texts,
    read_processed_splits,
    resolve_project_path,
    set_global_seed,
)


MODEL_NAME_FOR_REPORT = "phobert_base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT on clean UIT-VSFC data.")
    parser.add_argument(
        "--task",
        choices=["sentiment", "topic", "all"],
        required=True,
        help="Task to train. Use 'all' to train both sentiment and topic sequentially.",
    )
    parser.add_argument(
        "--config",
        default="configs/03_phobert_training_config.yaml",
        help="Path to Stage 3 YAML config.",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Project root directory.",
    )
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def compute_metrics_factory(labels: list[str]):
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        pred_ids = np.argmax(logits, axis=1)

        return {
            "accuracy": accuracy_score(label_ids, pred_ids),
            "macro_f1": f1_score(label_ids, pred_ids, average="macro", zero_division=0),
            "weighted_f1": f1_score(label_ids, pred_ids, average="weighted", zero_division=0),
        }

    return compute_metrics


def upsert_csv_rows(
    csv_path: Path,
    new_rows: list[dict[str, Any]],
    key_columns: list[str],
) -> pd.DataFrame:
    """Insert or replace rows in a CSV using key columns."""
    new_df = pd.DataFrame(new_rows)

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        if not old_df.empty:
            old_key = old_df[key_columns].astype(str).agg("||".join, axis=1)
            new_key = new_df[key_columns].astype(str).agg("||".join, axis=1)
            old_df = old_df[~old_key.isin(set(new_key))]
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
    else:
        combined_df = new_df

    combined_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return combined_df


def save_confusion_matrix_plot(cm_df: pd.DataFrame, title: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_df.values)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(cm_df.index)

    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            ax.text(j, i, str(cm_df.iloc[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def save_training_curve(history_df: pd.DataFrame, task: str, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if history_df.empty or "epoch" not in history_df.columns:
        return

    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    has_curve = False

    train_loss_df = history_df.dropna(subset=["loss"]) if "loss" in history_df.columns else pd.DataFrame()
    eval_loss_df = history_df.dropna(subset=["eval_loss"]) if "eval_loss" in history_df.columns else pd.DataFrame()

    if not train_loss_df.empty:
        plt.plot(train_loss_df["epoch"], train_loss_df["loss"], marker="o", label="train_loss")
        has_curve = True

    if not eval_loss_df.empty:
        plt.plot(eval_loss_df["epoch"], eval_loss_df["eval_loss"], marker="o", label="eval_loss")
        has_curve = True

    if not has_curve:
        plt.close()
        return

    plt.title(f"PhoBERT training loss curve - {task}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    output_path = figures_dir / f"03_phobert_training_loss_curve_{task}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_split(
    trainer,
    split_name: str,
    df: pd.DataFrame,
    dataset,
    task: str,
    label_col: str,
    labels: list[str],
    id2label: dict[int, str],
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Predict and export metrics for one split."""
    prediction_output = trainer.predict(dataset)
    logits = prediction_output.predictions
    pred_ids = np.argmax(logits, axis=1)
    probabilities = softmax(logits)
    confidences = probabilities.max(axis=1)

    y_true = df[label_col].astype(str).tolist()
    y_pred = [id2label[int(idx)] for idx in pred_ids]

    metrics = {
        "task": task,
        "model_name": MODEL_NAME_FOR_REPORT,
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "num_eval_samples": len(df),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    report_df.insert(0, "task", task)
    report_df.insert(1, "model_name", MODEL_NAME_FOR_REPORT)
    report_df.insert(2, "split", split_name)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    cm_csv_path = tables_dir / f"03_confusion_matrix_phobert_{task}_{split_name}.csv"
    cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

    cm_fig_path = figures_dir / f"03_phobert_{task}_confusion_matrix_{split_name}.png"
    save_confusion_matrix_plot(
        cm_df=cm_df,
        title=f"PhoBERT - {task} - {split_name}",
        output_path=cm_fig_path,
    )

    predictions_df = pd.DataFrame({
        "id": df["id"].tolist() if "id" in df.columns else list(range(len(df))),
        "text": df["text"].tolist(),
        "true_label": y_true,
        "pred_label": y_pred,
        "confidence": confidences,
    })

    predictions_path = tables_dir / f"03_phobert_{task}_{split_name}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    return metrics, report_df


def generate_comparison_and_report(
    project_root: Path,
    tables_dir: Path,
    notes_dir: Path,
    figures_dir: Path,
    config: dict[str, Any],
) -> None:
    """Generate Stage 3 report and PhoBERT-vs-baseline comparison when possible."""
    phobert_summary_path = tables_dir / "03_phobert_results_summary.csv"
    if not phobert_summary_path.exists():
        return

    phobert_summary = pd.read_csv(phobert_summary_path)

    baseline_summary_path = tables_dir / "02_baseline_results_summary.csv"
    best_baseline_path = tables_dir / "02_best_baseline_by_dev_macro_f1.csv"

    comparison_rows = []

    if baseline_summary_path.exists() and best_baseline_path.exists():
        baseline_summary = pd.read_csv(baseline_summary_path)
        best_baseline = pd.read_csv(best_baseline_path)

        for task in sorted(phobert_summary["task"].unique()):
            phobert_test = phobert_summary[
                (phobert_summary["task"] == task)
                & (phobert_summary["split"] == "test")
            ]

            best_task = best_baseline[best_baseline["task"] == task]

            if phobert_test.empty or best_task.empty:
                continue

            best_model_name = best_task.iloc[0]["model_name"]

            baseline_test = baseline_summary[
                (baseline_summary["task"] == task)
                & (baseline_summary["model_name"] == best_model_name)
                & (baseline_summary["split"] == "test")
            ]

            if baseline_test.empty:
                continue

            ph_row = phobert_test.iloc[0]
            base_row = baseline_test.iloc[0]

            comparison_rows.append({
                "task": task,
                "best_baseline_model": best_model_name,
                "baseline_test_accuracy": base_row["accuracy"],
                "baseline_test_macro_f1": base_row["macro_f1"],
                "baseline_test_weighted_f1": base_row["weighted_f1"],
                "phobert_test_accuracy": ph_row["accuracy"],
                "phobert_test_macro_f1": ph_row["macro_f1"],
                "phobert_test_weighted_f1": ph_row["weighted_f1"],
                "delta_macro_f1": ph_row["macro_f1"] - base_row["macro_f1"],
                "delta_accuracy": ph_row["accuracy"] - base_row["accuracy"],
            })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = tables_dir / "03_phobert_vs_baseline_comparison.csv"

    if not comparison_df.empty:
        comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
        save_comparison_plot(comparison_df, figures_dir)

    report_lines = []
    report_lines.append("# PhoBERT Clean Fine-tuning Report\n")
    report_lines.append(f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`")
    report_lines.append("- Stage: `03_phobert_clean_finetuning`")
    report_lines.append("- Dataset: processed UIT-VSFC splits from Stage 1")
    report_lines.append("- Main metric: Macro-F1")
    report_lines.append("- Model selection split: dev")
    report_lines.append("- Final reporting split: test\n")

    report_lines.append("## Model and preprocessing\n")
    report_lines.append(f"- Model name: `{config['model']['model_name']}`")
    report_lines.append(f"- Max length: `{config['model']['max_length']}`")
    report_lines.append(
        f"- Word segmentation enabled: `{config.get('preprocessing', {}).get('word_segmentation_enabled', False)}`"
    )
    report_lines.append(
        f"- Word segmentation method: `{config.get('preprocessing', {}).get('word_segmentation_method', 'none')}`\n"
    )

    report_lines.append("## Hyperparameters\n")
    hyperparam_df = pd.DataFrame([
        {"parameter": key, "value": value}
        for key, value in config["training"].items()
    ])
    report_lines.append(dataframe_to_markdown(hyperparam_df))
    report_lines.append("\n")

    report_lines.append("## PhoBERT results\n")
    display_summary = phobert_summary.copy()
    for col in ["accuracy", "macro_f1", "weighted_f1", "train_time_sec", "predict_time_sec"]:
        if col in display_summary.columns:
            display_summary[col] = display_summary[col].round(4)
    report_lines.append(dataframe_to_markdown(display_summary))
    report_lines.append("\n")

    report_lines.append("## PhoBERT vs best Stage 2 baseline\n")
    if comparison_df.empty:
        report_lines.append("_Comparison is not available yet. Ensure Stage 2 outputs exist._\n")
    else:
        display_comp = comparison_df.copy()
        for col in display_comp.columns:
            if col.endswith("f1") or col.endswith("accuracy"):
                display_comp[col] = display_comp[col].round(4)
        if "delta_macro_f1" in display_comp.columns:
            display_comp["delta_macro_f1"] = display_comp["delta_macro_f1"].round(4)
        if "delta_accuracy" in display_comp.columns:
            display_comp["delta_accuracy"] = display_comp["delta_accuracy"].round(4)
        report_lines.append(dataframe_to_markdown(display_comp))
        report_lines.append("\n")

    report_lines.append("## Notes\n")
    report_lines.append(
        "- Test results are used only for final reporting, not for model selection.\n"
        "- Macro-F1 remains the primary metric because both tasks are class-imbalanced.\n"
        "- No noisy-data evaluation is performed in Stage 3.\n"
        "- The current default config does not apply external Vietnamese word segmentation; this should be reported as a controlled implementation choice and revisited in the segmentation/tokenization analysis stage.\n"
    )

    notes_dir.mkdir(parents=True, exist_ok=True)
    report_path = notes_dir / "03_phobert_clean_finetuning_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def save_comparison_plot(comparison_df: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if comparison_df.empty:
        return

    plot_df = comparison_df[["task", "baseline_test_macro_f1", "phobert_test_macro_f1"]].copy()
    plot_df = plot_df.set_index("task")

    ax = plot_df.plot(kind="bar", figsize=(9, 5))
    ax.set_title("PhoBERT vs Best Stage 2 Baseline - Test Macro-F1")
    ax.set_xlabel("Task")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "03_phobert_vs_baseline_macro_f1.png"
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_task(task: str, config: dict[str, Any], project_root: Path) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    set_global_seed(int(config["training"]["seed"]))

    project_config = config["project"]

    data_dir = resolve_project_path(project_root, project_config["data_dir"])
    label_mapping_path = resolve_project_path(project_root, project_config["label_mapping_path"])
    tables_dir = resolve_project_path(project_root, project_config["reports_tables_dir"])
    figures_dir = resolve_project_path(project_root, project_config["reports_figures_dir"])
    notes_dir = resolve_project_path(project_root, project_config["reports_notes_dir"])
    models_dir = resolve_project_path(project_root, project_config["models_dir"])

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    task_config = config["tasks"][task]
    text_col = task_config["text_column"]
    label_col = task_config["label_column"]
    output_subdir = task_config["output_subdir"]

    labels, label2id, id2label = load_label_info(label_mapping_path, task)

    frames = read_processed_splits(data_dir)

    for split, df in frames.items():
        for col in [text_col, label_col]:
            if col not in df.columns:
                raise KeyError(f"Missing column '{col}' in {split} split.")

    preprocessing_config = config.get("preprocessing", {})
    train_texts = prepare_texts(frames["train"][text_col], preprocessing_config)
    dev_texts = prepare_texts(frames["dev"][text_col], preprocessing_config)
    test_texts = prepare_texts(frames["test"][text_col], preprocessing_config)

    train_labels = frames["train"][label_col].astype(str).tolist()
    dev_labels = frames["dev"][label_col].astype(str).tolist()
    test_labels = frames["test"][label_col].astype(str).tolist()

    model_name = config["model"]["model_name"]
    max_length = int(config["model"]["max_length"])

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    train_dataset = PhoBERTClassificationDataset(
        texts=train_texts,
        labels=train_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )

    dev_dataset = PhoBERTClassificationDataset(
        texts=dev_texts,
        labels=dev_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )

    test_dataset = PhoBERTClassificationDataset(
        texts=test_texts,
        labels=test_labels,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )

    task_model_dir = models_dir / output_subdir
    trainer_output_dir = task_model_dir / "trainer_output"
    best_model_dir = task_model_dir / "best_model"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label={str(k): v for k, v in id2label.items()},
        label2id=label2id,
    )

    fp16_requested = bool(config["training"].get("fp16", False))
    fp16_enabled = fp16_requested and torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(trainer_output_dir),
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=float(config["training"]["num_train_epochs"]),
        per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config["training"]["per_device_eval_batch_size"]),
        weight_decay=float(config["training"]["weight_decay"]),
        warmup_ratio=float(config["training"]["warmup_ratio"]),
        logging_steps=int(config["training"]["logging_steps"]),
        evaluation_strategy=str(config["training"]["evaluation_strategy"]),
        save_strategy=str(config["training"]["save_strategy"]),
        load_best_model_at_end=bool(config["training"]["load_best_model_at_end"]),
        metric_for_best_model=str(config["training"]["metric_for_best_model"]),
        greater_is_better=bool(config["training"]["greater_is_better"]),
        fp16=fp16_enabled,
        report_to=[] if str(config["training"].get("report_to", "none")).lower() == "none" else None,
        save_total_limit=2,
        seed=int(config["training"]["seed"]),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_factory(labels),
    )

    start_train = time.perf_counter()
    train_result = trainer.train()
    train_time_sec = time.perf_counter() - start_train

    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    history_df = pd.DataFrame(trainer.state.log_history)
    history_df.insert(0, "task", task)
    history_path = tables_dir / f"03_phobert_training_history_{task}.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    save_training_curve(history_df, task, figures_dir)

    report_frames = []
    summary_rows = []

    split_datasets = {
        "dev": (frames["dev"], dev_dataset),
        "test": (frames["test"], test_dataset),
    }

    for split_name, (split_df, split_dataset) in split_datasets.items():
        start_predict = time.perf_counter()
        metrics, report_df = evaluate_split(
            trainer=trainer,
            split_name=split_name,
            df=split_df,
            dataset=split_dataset,
            task=task,
            label_col=label_col,
            labels=labels,
            id2label=id2label,
            tables_dir=tables_dir,
            figures_dir=figures_dir,
        )
        predict_time_sec = time.perf_counter() - start_predict

        metrics["train_time_sec"] = train_time_sec
        metrics["predict_time_sec"] = predict_time_sec
        metrics["num_train_samples"] = len(frames["train"])
        metrics["model_path"] = str(best_model_dir)

        summary_rows.append(metrics)
        report_frames.append(report_df)

    summary_path = tables_dir / "03_phobert_results_summary.csv"
    combined_summary = upsert_csv_rows(
        csv_path=summary_path,
        new_rows=summary_rows,
        key_columns=["task", "model_name", "split"],
    )

    task_report_df = pd.concat(report_frames, ignore_index=True)
    task_report_path = tables_dir / f"03_phobert_{task}_classification_report.csv"
    task_report_df.to_csv(task_report_path, index=False, encoding="utf-8-sig")

    all_report_path = tables_dir / "03_phobert_classification_reports.csv"
    upsert_csv_rows(
        csv_path=all_report_path,
        new_rows=task_report_df.to_dict(orient="records"),
        key_columns=["task", "model_name", "split", "label"],
    )

    metadata = {
        "task": task,
        "model_name": model_name,
        "labels": labels,
        "label2id": label2id,
        "id2label": id2label,
        "max_length": max_length,
        "preprocessing": preprocessing_config,
        "training": config["training"],
        "train_result": train_result.metrics,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "best_model_dir": str(best_model_dir),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    metadata_path = tables_dir / f"03_phobert_{task}_metadata.json"
    metadata_path.write_text(
        pd.Series(metadata).to_json(force_ascii=False, indent=2),
        encoding="utf-8",
    )

    generate_comparison_and_report(
        project_root=project_root,
        tables_dir=tables_dir,
        notes_dir=notes_dir,
        figures_dir=figures_dir,
        config=config,
    )

    print(f"Finished task: {task}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved classification report: {task_report_path}")
    print(f"Saved model: {best_model_dir}")


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = resolve_project_path(project_root, args.config)
    config = load_yaml(config_path)

    tasks_to_run = ["sentiment", "topic"] if args.task == "all" else [args.task]

    for task in tasks_to_run:
        if task not in config["tasks"]:
            raise KeyError(f"Task is not configured: {task}")

        run_task(task=task, config=config, project_root=project_root)


if __name__ == "__main__":
    main()
