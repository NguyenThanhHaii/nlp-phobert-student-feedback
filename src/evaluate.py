"""Reusable evaluation helpers for model experiments."""

from __future__ import annotations

from pathlib import Path
import time

import joblib
import pandas as pd

from src.metrics import evaluate_predictions, plot_confusion_matrix


def _prefix_filename(filename: str, output_prefix: str | None = None) -> str:
    """Add a stage prefix to an output filename when requested."""
    if output_prefix is None or str(output_prefix).strip() == "":
        return filename

    clean_prefix = str(output_prefix).strip().rstrip("_")
    return f"{clean_prefix}_{filename}"


def train_and_evaluate_model(
    model,
    model_name: str,
    task: str,
    train_df: pd.DataFrame,
    eval_frames: dict[str, pd.DataFrame],
    text_col: str,
    label_col: str,
    labels: list[str],
    models_dir: Path,
    reports_tables_dir: Path,
    reports_figures_dir: Path,
    output_prefix: str | None = None,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Train a model and evaluate it on multiple splits.

    Args:
        model: sklearn-compatible estimator.
        model_name: Stable model name for reports and files.
        task: Task name, e.g. "sentiment" or "topic".
        train_df: Training DataFrame.
        eval_frames: Mapping from split name to evaluation DataFrame.
        text_col: Text column name.
        label_col: Label column name.
        labels: Ordered label list.
        models_dir: Base directory for saved models.
        reports_tables_dir: Directory for CSV outputs.
        reports_figures_dir: Directory for figures.
        output_prefix: Optional stage prefix for generated report/figure filenames.

    Returns:
        summary_rows: list of flat metric rows.
        report_frames: list of classification report DataFrames.
    """
    task_model_dir = models_dir / task
    task_model_dir.mkdir(parents=True, exist_ok=True)

    reports_tables_dir.mkdir(parents=True, exist_ok=True)
    reports_figures_dir.mkdir(parents=True, exist_ok=True)

    x_train = train_df[text_col].tolist()
    y_train = train_df[label_col].tolist()

    start_train = time.perf_counter()
    model.fit(x_train, y_train)
    train_time_sec = time.perf_counter() - start_train

    model_path = task_model_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    summary_rows = []
    report_frames = []

    for split, eval_df in eval_frames.items():
        x_eval = eval_df[text_col].tolist()
        y_eval = eval_df[label_col].tolist()

        start_predict = time.perf_counter()
        y_pred = model.predict(x_eval)
        predict_time_sec = time.perf_counter() - start_predict

        metrics, report_df, cm_df = evaluate_predictions(
            y_true=y_eval,
            y_pred=y_pred,
            labels=labels,
        )

        summary_rows.append({
            "task": task,
            "model_name": model_name,
            "split": split,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "train_time_sec": train_time_sec,
            "predict_time_sec": predict_time_sec,
            "num_train_samples": len(train_df),
            "num_eval_samples": len(eval_df),
            "model_path": str(model_path),
        })

        report_df.insert(0, "task", task)
        report_df.insert(1, "model_name", model_name)
        report_df.insert(2, "split", split)
        report_frames.append(report_df)

        cm_filename = _prefix_filename(
            f"confusion_matrix_{task}_{model_name}_{split}.csv",
            output_prefix=output_prefix,
        )
        cm_path = reports_tables_dir / cm_filename
        cm_df.to_csv(cm_path, encoding="utf-8-sig")

        fig_filename = _prefix_filename(
            f"confusion_matrix_{task}_{model_name}_{split}.png",
            output_prefix=output_prefix,
        )
        fig_path = reports_figures_dir / fig_filename
        plot_confusion_matrix(
            cm_df=cm_df,
            title=f"{task.title()} - {model_name} - {split}",
            output_path=fig_path,
        )

    return summary_rows, report_frames
