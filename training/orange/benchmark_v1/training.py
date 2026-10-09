"""Approved two-phase training bound to verified Binary V1 contexts."""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import EXPECTED_COUNTS, sha256_file
from .context import (
    IsolatedRunDirectory,
    VerifiedCheckpoint,
    VerifiedClassWeights,
    VerifiedRunContext,
    VerifiedTrainDataset,
    VerifiedTrainPartition,
    VerifiedValidationDataset,
    _CHECKPOINT_SEAL,
    _WEIGHT_SEAL,
    context_fingerprint,
)
from .metrics import calculate_metrics


class TrainingContractError(RuntimeError):
    pass


@dataclass
class GlobalBestState:
    best_val_loss: float | None = None
    best_phase: str | None = None
    best_epoch_in_phase: int | None = None
    best_global_epoch: int | None = None

    def observe(self, *, phase: str, epoch_in_phase: int, global_epoch: int, val_loss: float) -> bool:
        value = float(val_loss)
        if phase not in {"P1", "P2"} or epoch_in_phase < 1 or global_epoch < 1:
            raise ValueError("Invalid phase/epoch metadata.")
        if not isfinite(value):
            raise ValueError("val_loss must be finite.")
        if self.best_val_loss is None or value < self.best_val_loss:
            self.best_val_loss = value
            self.best_phase = phase
            self.best_epoch_in_phase = int(epoch_in_phase)
            self.best_global_epoch = int(global_epoch)
            return True
        return False


@dataclass(frozen=True)
class TrainingResult:
    global_best: GlobalBestState
    best_checkpoint: VerifiedCheckpoint
    last_checkpoint: VerifiedCheckpoint
    phase1_history: Mapping[str, list[float]]
    phase2_history: Mapping[str, list[float]]
    phase_metrics: tuple[Mapping[str, Any], ...]
    class_weights: VerifiedClassWeights


def select_global_best(rows: Iterable[Mapping[str, object]]) -> GlobalBestState:
    state = GlobalBestState()
    for row in rows:
        state.observe(
            phase=str(row["phase"]), epoch_in_phase=int(row["epoch_in_phase"]),
            global_epoch=int(row["global_epoch"]), val_loss=float(row["val_loss"]),
        )
    if state.best_val_loss is None:
        raise ValueError("At least one epoch row is required.")
    return state


def compute_class_weights(partition: VerifiedTrainPartition) -> VerifiedClassWeights:
    if not isinstance(partition, VerifiedTrainPartition):
        raise TypeError("Class weights require a verified DEV_TRAIN partition.")
    counts = {0: 0, 1: 0}
    for row in partition.records:
        counts[row.label_id] += 1
    expected = EXPECTED_COUNTS["DEV_TRAIN"]
    if counts != {0: expected["GOOD"], 1: expected["BAD"]}:
        raise TrainingContractError(f"Unexpected DEV_TRAIN label counts: {counts}")
    total = len(partition.records)
    values = {label: total / (2.0 * count) for label, count in counts.items()}
    return VerifiedClassWeights(values, partition.manifest_sha256, total, _WEIGHT_SEAL)


def build_model(config) -> tuple[Any, Any]:
    from .context import VerifiedConfig

    if not isinstance(config, VerifiedConfig):
        raise TypeError("Model construction requires VerifiedConfig.")
    import tensorflow as tf

    tf.keras.mixed_precision.set_global_policy("float32")
    from orange.model import build_mobilenetv3_small

    return build_mobilenetv3_small(
        image_size=int(config.values["data"]["image_size"]),
        dropout=float(config.values["model"]["dropout"]),
    )


def _compile(model, learning_rate: float) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision_bad"),
            tf.keras.metrics.Recall(name="recall_bad"),
            tf.keras.metrics.AUC(name="roc_auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )


def configure_phase1(model, backbone, learning_rate: float) -> None:
    backbone.trainable = False
    _compile(model, learning_rate)


def configure_phase2(model, backbone, learning_rate: float) -> None:
    import tensorflow as tf

    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    _compile(model, learning_rate)  # Constructs a new optimizer.


def _callbacks(
    *, state: GlobalBestState, phase: str, global_epoch_offset: int, best_path: Path,
    patience: int, validation: VerifiedValidationDataset, phase_rows: list[dict[str, Any]],
):
    import tensorflow as tf

    class EvidenceAndCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if "val_loss" not in logs:
                raise TrainingContractError("val_loss missing; refusing checkpoint selection.")
            probabilities = self.model.predict(validation.dataset, verbose=0).reshape(-1).tolist()
            labels: list[int] = []
            for _, batch_labels in validation.dataset:
                labels.extend(int(value) for value in batch_labels.numpy().reshape(-1).tolist())
            metrics = calculate_metrics(labels, probabilities, threshold=0.5)
            good = [p for y, p in zip(labels, probabilities) if y == 0]
            bad = [p for y, p in zip(labels, probabilities) if y == 1]
            good_median, bad_median = statistics.median(good), statistics.median(bad)
            global_epoch = global_epoch_offset + epoch + 1
            phase_rows.append({
                "phase": phase, "epoch_in_phase": epoch + 1, "global_epoch": global_epoch,
                "train_loss": float(logs["loss"]), "train_accuracy": float(logs["accuracy"]),
                "val_loss": float(logs["val_loss"]), "val_accuracy": float(logs["val_accuracy"]),
                "roc_auc": float(metrics["roc_auc"]), "pr_auc": float(metrics["pr_auc"]),
                "pbad_mean": statistics.fmean(probabilities),
                "pbad_std": statistics.pstdev(probabilities),
                "good_pbad_median": good_median, "bad_pbad_median": bad_median,
                "median_separation": bad_median - good_median,
                "fn_bad_0_5": metrics["fn_bad"], "fp_good_0_5": metrics["fp_good"],
                "fpr_good_0_5": metrics["fpr_good"],
                "collapse_flag": "PREDICTION_COLLAPSE_SUSPECTED" if statistics.pstdev(probabilities) < 1e-3 else "OK",
            })
            if state.observe(
                phase=phase, epoch_in_phase=epoch + 1, global_epoch=global_epoch,
                val_loss=float(logs["val_loss"]),
            ):
                self.model.save(best_path)

    return [
        EvidenceAndCheckpoint(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=patience, restore_best_weights=False,
        ),
    ]


def _validate_runtime_bindings(
    context: VerifiedRunContext,
    run_directory: IsolatedRunDirectory,
    train: VerifiedTrainDataset,
    validation: VerifiedValidationDataset,
    weights: VerifiedClassWeights,
) -> None:
    if not all((
        isinstance(context, VerifiedRunContext),
        isinstance(run_directory, IsolatedRunDirectory),
        isinstance(train, VerifiedTrainDataset),
        isinstance(validation, VerifiedValidationDataset),
        isinstance(weights, VerifiedClassWeights),
    )):
        raise TypeError("Training requires verifier-issued context, directory, datasets, and weights.")
    fingerprint = context_fingerprint(
        context.config.sha256, context.config.protocol_sha256,
        context.manifest.sha256, context.approval.sha256,
    )
    if run_directory.run_id != context.run_id or run_directory.context_fingerprint != fingerprint:
        raise TrainingContractError("Run directory is not bound to this context.")
    expected_run_path = (context.config.repo_root / "outputs_v1" / context.run_id).resolve()
    if run_directory.path.resolve() != expected_run_path:
        raise TrainingContractError("Run directory path is not canonical for this context.")
    if any(value != context.manifest.sha256 for value in (
        train.manifest_sha256, validation.manifest_sha256, weights.manifest_sha256
    )):
        raise TrainingContractError("Dataset/class-weight manifest binding mismatch.")
    if train.sampling_strategy != "NONE":
        raise TrainingContractError("Double balancing is forbidden.")
    expected_batch = int(context.config.values["data"]["batch_size"])
    expected_image_size = int(context.config.values["data"]["image_size"])
    if (
        train.config_sha256 != context.config.sha256
        or validation.config_sha256 != context.config.sha256
        or train.batch_size != expected_batch
        or validation.batch_size != expected_batch
        or train.image_size != expected_image_size
        or validation.image_size != expected_image_size
    ):
        raise TrainingContractError("Dataset/config preprocessing binding mismatch.")
    if (train.sample_count != EXPECTED_COUNTS["DEV_TRAIN"]["total"]
            or validation.sample_count != EXPECTED_COUNTS["DEV_VALIDATION"]["total"]):
        raise TrainingContractError("Unexpected verified dataset sizes.")
    expected_counts = EXPECTED_COUNTS["DEV_TRAIN"]
    expected_weights = {
        0: expected_counts["total"] / (2.0 * expected_counts["GOOD"]),
        1: expected_counts["total"] / (2.0 * expected_counts["BAD"]),
    }
    if weights.train_count != expected_counts["total"] or any(
        abs(float(weights.values[label]) - expected_weights[label]) > 1e-12
        for label in (0, 1)
    ):
        raise TrainingContractError("Class weights do not match verified DEV_TRAIN counts.")


def _fit_two_phase(
    *, context: VerifiedRunContext, run_directory: IsolatedRunDirectory,
    model, backbone, train: VerifiedTrainDataset,
    validation: VerifiedValidationDataset, class_weights: VerifiedClassWeights,
) -> TrainingResult:
    """Internal training helper; execute_approved_run() is the public entrypoint."""
    _validate_runtime_bindings(context, run_directory, train, validation, class_weights)
    config = context.config.values
    p1, p2 = config["training"]["phase1"], config["training"]["phase2"]
    seed = int(config["project"]["seed"])
    random.seed(seed)
    import numpy as np
    import tensorflow as tf

    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    best_path = run_directory.path / "best_model.keras"
    last_path = run_directory.path / "last_model.keras"
    if best_path.exists() or last_path.exists():
        raise FileExistsError("Refusing to overwrite existing run checkpoints.")

    state = GlobalBestState()
    phase_rows: list[dict[str, Any]] = []
    configure_phase1(model, backbone, float(p1["learning_rate"]))
    p1_history = model.fit(
        train.dataset, validation_data=validation.dataset, epochs=int(p1["max_epochs"]),
        class_weight=dict(class_weights.values),
        callbacks=_callbacks(
            state=state, phase="P1", global_epoch_offset=0, best_path=best_path,
            patience=int(p1["early_stopping_patience"]), validation=validation,
            phase_rows=phase_rows,
        ), verbose=1,
    )
    p1_executed = len(p1_history.epoch)
    configure_phase2(model, backbone, float(p2["learning_rate"]))
    p2_history = model.fit(
        train.dataset, validation_data=validation.dataset, epochs=int(p2["max_epochs"]),
        class_weight=dict(class_weights.values),
        callbacks=_callbacks(
            state=state, phase="P2", global_epoch_offset=p1_executed, best_path=best_path,
            patience=int(p2["early_stopping_patience"]), validation=validation,
            phase_rows=phase_rows,
        ), verbose=1,
    )
    if not best_path.is_file():
        raise TrainingContractError("Global-best checkpoint was not created.")
    model.save(last_path)
    fingerprint = run_directory.context_fingerprint
    best = VerifiedCheckpoint(
        best_path, sha256_file(best_path), "GLOBAL_BEST", context.run_id,
        fingerprint, _CHECKPOINT_SEAL,
    )
    last = VerifiedCheckpoint(
        last_path, sha256_file(last_path), "LAST", context.run_id,
        fingerprint, _CHECKPOINT_SEAL,
    )
    return TrainingResult(
        global_best=state, best_checkpoint=best, last_checkpoint=last,
        phase1_history=p1_history.history, phase2_history=p2_history.history,
        phase_metrics=tuple(phase_rows), class_weights=class_weights,
    )
