"""Validation evaluation that can only reload a verified global-best checkpoint."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import statistics
from typing import Any, Mapping

from .config import sha256_file
from .context import (
    IsolatedRunDirectory,
    VerifiedCheckpoint,
    VerifiedRunContext,
    VerifiedValidationDataset,
)
from .metrics import calculate_metrics, threshold_sweep


class EvaluationContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidationEvidence:
    run_id: str
    checkpoint_sha256: str
    predictions: tuple[Mapping[str, Any], ...]
    fixed_metrics: Mapping[str, Any]
    threshold_rows: tuple[Mapping[str, Any], ...]
    per_original_class: tuple[Mapping[str, Any], ...]


def _validate_best_checkpoint(
    context: VerifiedRunContext,
    run_directory: IsolatedRunDirectory,
    checkpoint: VerifiedCheckpoint,
) -> None:
    if not all((
        isinstance(context, VerifiedRunContext),
        isinstance(run_directory, IsolatedRunDirectory),
        isinstance(checkpoint, VerifiedCheckpoint),
    )):
        raise TypeError("Evaluation requires verifier-issued context, directory, and checkpoint.")
    if checkpoint.kind != "GLOBAL_BEST":
        raise EvaluationContractError("Only GLOBAL_BEST may be evaluated.")
    if checkpoint.run_id != context.run_id or run_directory.run_id != context.run_id:
        raise EvaluationContractError("Run ID mismatch.")
    if checkpoint.context_fingerprint != run_directory.context_fingerprint:
        raise EvaluationContractError("Checkpoint context fingerprint mismatch.")
    expected_path = (run_directory.path / "best_model.keras").resolve()
    if checkpoint.path.resolve() != expected_path or not expected_path.is_file():
        raise EvaluationContractError("Global-best checkpoint path is invalid.")
    if sha256_file(expected_path) != checkpoint.sha256:
        raise EvaluationContractError("Global-best checkpoint SHA-256 mismatch.")


def evaluate_validation(
    *, context: VerifiedRunContext, run_directory: IsolatedRunDirectory,
    validation: VerifiedValidationDataset, checkpoint: VerifiedCheckpoint,
) -> ValidationEvidence:
    _validate_best_checkpoint(context, run_directory, checkpoint)
    if not isinstance(validation, VerifiedValidationDataset):
        raise TypeError("Evaluation requires VerifiedValidationDataset.")
    expected_count = int(context.manifest.counts["DEV_VALIDATION"]["total"])
    if validation.manifest_sha256 != context.manifest.sha256 or validation.sample_count != expected_count:
        raise EvaluationContractError("Validation dataset binding/count mismatch.")
    import tensorflow as tf

    model = tf.keras.models.load_model(checkpoint.path, compile=False)
    probabilities = model.predict(validation.dataset, verbose=1).reshape(-1).tolist()
    labels: list[int] = []
    for _, batch_labels in validation.dataset:
        labels.extend(int(value) for value in batch_labels.numpy().reshape(-1).tolist())
    records = context.manifest.validation.records
    if len(probabilities) != len(labels) or len(labels) != len(records):
        raise EvaluationContractError("Prediction/label/manifest length mismatch.")
    label_tensor = tf.reshape(tf.convert_to_tensor(labels, dtype=tf.float32), (-1, 1))
    probability_tensor = tf.reshape(
        tf.convert_to_tensor(probabilities, dtype=tf.float32), (-1, 1)
    )
    val_loss = float(tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(label_tensor, probability_tensor)
    ).numpy())
    metrics = dict(calculate_metrics(labels, probabilities, threshold=0.5))
    metrics["val_loss"] = val_loss
    prediction_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record, label, probability in zip(records, labels, probabilities):
        prediction_rows.append({
            "run_id": context.run_id,
            "relative_path": record.relative_path,
            "original_class": record.original_class,
            "split": "DEV_VALIDATION",
            "true_label": label,
            "p_BAD": probability,
            "p_GOOD": 1.0 - probability,
            "predicted_label_at_0_5": int(probability >= 0.5),
            "checkpoint_sha256": checkpoint.sha256,
        })
        grouped[record.original_class].append((label, probability))
    class_rows = []
    for original_class, values in sorted(grouped.items()):
        class_labels = [item[0] for item in values]
        class_probabilities = [item[1] for item in values]
        predicted = [int(value >= 0.5) for value in class_probabilities]
        class_rows.append({
            "original_class": original_class,
            "total": len(values),
            "correct": sum(a == b for a, b in zip(class_labels, predicted)),
            "incorrect": sum(a != b for a, b in zip(class_labels, predicted)),
            "mean_p_BAD": sum(class_probabilities) / len(class_probabilities),
            "median_p_BAD": statistics.median(class_probabilities),
            "bad_recall": (
                sum(value >= 0.5 for value in class_probabilities) / len(class_probabilities)
                if all(label == 1 for label in class_labels) else None
            ),
        })
    return ValidationEvidence(
        run_id=context.run_id,
        checkpoint_sha256=checkpoint.sha256,
        predictions=tuple(prediction_rows),
        fixed_metrics=metrics,
        threshold_rows=tuple(threshold_sweep(labels, probabilities)),
        per_original_class=tuple(class_rows),
    )
