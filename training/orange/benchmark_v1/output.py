"""Strict Binary V1 artifact schemas and cross-artifact validation."""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import sha256_file
from .context import VerifiedCheckpoint, VerifiedRunContext, context_fingerprint
from .metrics import calculate_metrics

RUN_STATUSES = {"PASS", "FAIL", "INCOMPLETE"}
GATE_STATUSES = {"PASS", "FAIL", "BLOCKED", "UNVERIFIED", "NOT_RUN"}
PHASE_METRIC_COLUMNS = (
    "phase", "epoch_in_phase", "global_epoch", "train_loss", "train_accuracy",
    "val_loss", "val_accuracy", "roc_auc", "pr_auc", "pbad_mean", "pbad_std",
    "good_pbad_median", "bad_pbad_median", "median_separation", "fn_bad_0_5",
    "fp_good_0_5", "fpr_good_0_5", "collapse_flag",
)
PREDICTION_COLUMNS = (
    "run_id", "relative_path", "original_class", "split", "true_label",
    "p_BAD", "p_GOOD", "predicted_label_at_0_5", "checkpoint_sha256",
)
FIXED_METRIC_FIELDS = {
    "threshold", "n", "val_loss", "accuracy", "precision_bad", "recall_bad",
    "f1_bad", "macro_f1", "roc_auc", "pr_auc", "brier", "fn_bad", "fp_good",
    "fpr_good", "confusion_matrix",
}
THRESHOLD_COLUMNS = (
    "threshold", "accuracy", "precision_bad", "recall_bad", "f1_bad",
    "macro_f1", "fn_bad", "fp_good", "fpr_good",
)
THRESHOLD_FIELDS = set(THRESHOLD_COLUMNS)
REQUIRED_RUN_ARTIFACTS = {
    "config_snapshot.yaml", "environment.json", "source_provenance.json",
    "data_integrity.json", "dataset_manifest_sha256.txt", "protocol_sha256.txt",
    "training_history.csv", "phase_metrics.csv", "model_summary.txt",
    "layer_inventory.csv", "best_model.keras", "last_model.keras",
    "global_best_metadata.json", "validation_predictions.csv",
    "validation_metrics_fixed_0_5.json", "validation_original_class_metrics.csv",
    "threshold_sweep.csv", "validation_threshold_diagnostic.json",
    "best_vs_last.json", "training_resource_summary.json", "run_status.json",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CLASS_TO_LABEL = {
    "fresh_orange": 0,
    "blackspotted_orange": 1,
    "cankered_orange": 1,
    "rotten_orange": 1,
}


class OutputContractError(ValueError):
    pass


def _assert_context_checkpoint(context: VerifiedRunContext, checkpoint: VerifiedCheckpoint) -> None:
    if not isinstance(context, VerifiedRunContext) or not isinstance(checkpoint, VerifiedCheckpoint):
        raise TypeError("Verified run context and checkpoint are required.")
    if checkpoint.kind != "GLOBAL_BEST" or checkpoint.run_id != context.run_id:
        raise OutputContractError("Output must be bound to the verified global-best checkpoint.")
    if not _HEX64.fullmatch(checkpoint.sha256):
        raise OutputContractError("Checkpoint SHA-256 is invalid.")
    expected_fingerprint = context_fingerprint(
        context.config.sha256, context.config.protocol_sha256,
        context.manifest.sha256, context.approval.sha256,
    )
    if checkpoint.context_fingerprint != expected_fingerprint:
        raise OutputContractError("Checkpoint/context fingerprint mismatch.")
    if checkpoint.path.name != "best_model.keras" or not checkpoint.path.is_file():
        raise OutputContractError("Global-best checkpoint file is missing or misnamed.")
    if sha256_file(checkpoint.path) != checkpoint.sha256:
        raise OutputContractError("Global-best checkpoint SHA-256 mismatch.")


def incomplete_run_status(
    *, context: VerifiedRunContext | None, reason: str, gate_status: str = "BLOCKED",
    execution_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if gate_status not in GATE_STATUSES - {"PASS"}:
        raise OutputContractError("Incomplete run gate status cannot be PASS.")
    evidence = dict(execution_evidence or {})
    holdout = evidence.get("holdout_access")
    if holdout is not None and set(holdout) != {"loaded", "predicted", "evaluated", "thresholded"}:
        raise OutputContractError("Malformed holdout execution evidence.")
    return {
        "status": "INCOMPLETE",
        "gate_status": gate_status,
        "stage": "PREFLIGHT" if context is None else context.stage,
        "backbone": "MobileNetV3-Small",
        "run_id": None if context is None else context.run_id,
        "source_commit": None if context is None else context.config.values["contract"]["reference_commit"],
        "dirty_tree": None,
        "frozen_protocol_sha256": None if context is None else context.config.protocol_sha256,
        "config_sha256": None if context is None else context.config.sha256,
        "manifest_sha256": None if context is None else context.manifest.sha256,
        "checkpoint_sha256": None,
        "training_completed": False,
        "holdout_access": holdout,
        "holdout_evidence_status": "NOT_RUN" if holdout is None else "INSTRUMENTED",
        "artifact_paths": {},
        "exception_approvals": [],
        "scientific_test_status": "NOT_AVAILABLE",
        "reason": reason,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def fixed_metrics_payload(
    context: VerifiedRunContext, checkpoint: VerifiedCheckpoint, metrics: Mapping[str, Any]
) -> dict[str, Any]:
    _assert_context_checkpoint(context, checkpoint)
    if not FIXED_METRIC_FIELDS.issubset(metrics):
        raise OutputContractError(f"Fixed metrics missing: {sorted(FIXED_METRIC_FIELDS - set(metrics))}")
    expected_count = int(context.manifest.counts["DEV_VALIDATION"]["total"])
    if float(metrics["threshold"]) != 0.5 or int(metrics["n"]) != expected_count:
        raise OutputContractError(
            f"Primary metrics require threshold 0.5 and n={expected_count}."
        )
    matrix = metrics["confusion_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 2 or any(len(row) != 2 for row in matrix):
        raise OutputContractError("Confusion matrix must be [[TN,FP],[FN,TP]].")
    if sum(int(value) for row in matrix for value in row) != expected_count:
        raise OutputContractError("Confusion matrix count mismatch.")
    for key in FIXED_METRIC_FIELDS - {"confusion_matrix"}:
        value = metrics[key]
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise OutputContractError(f"Fixed metric must be finite: {key}")
    protected = {key: metrics[key] for key in FIXED_METRIC_FIELDS}
    return {
        "schema_version": "NCKH_BASELINE_V1",
        "run_id": context.run_id,
        "status": "DEVELOPMENT_VALIDATION_ONLY",
        "checkpoint_kind": "GLOBAL_BEST",
        "checkpoint_sha256": checkpoint.sha256,
        **protected,
    }


def data_integrity_payload(context: VerifiedRunContext, file_evidence: Mapping[str, str]) -> dict[str, Any]:
    required = {"train_file_verification", "validation_file_verification"}
    if set(file_evidence) != required or any(value not in GATE_STATUSES for value in file_evidence.values()):
        raise OutputContractError("Explicit train/validation file evidence is required.")
    return {
        "schema_version": "NCKH_BASELINE_V1",
        "data_status": "LEGACY_DEV_INTERNAL_NOT_INDEPENDENT",
        "source_manifest_sha256": context.manifest.source_manifest_sha256,
        "portable_manifest_sha256": context.manifest.sha256,
        "counts": {key: dict(value) for key, value in context.manifest.counts.items()},
        **dict(file_evidence),
        "holdout_access": None,
        "holdout_evidence_status": "NOT_RUN",
        "green_orange_included": False,
        "membership_drift_count": 0,
        "label_drift_count": 0,
    }


def validate_prediction_rows(
    rows: Iterable[Mapping[str, Any]], context: VerifiedRunContext,
    checkpoint: VerifiedCheckpoint,
) -> tuple[dict[str, Any], ...]:
    _assert_context_checkpoint(context, checkpoint)
    materialized = tuple(dict(row) for row in rows)
    expected_records = {
        record.relative_path: (record.original_class, record.label_id)
        for record in context.manifest.validation.records
    }
    if len(materialized) != len(expected_records):
        raise OutputContractError(
            f"Validation predictions must contain exactly {len(expected_records)} rows."
        )
    seen: set[str] = set()
    labels = {0: 0, 1: 0}
    for row in materialized:
        missing = set(PREDICTION_COLUMNS) - set(row)
        if missing:
            raise OutputContractError(f"Prediction fields missing: {sorted(missing)}")
        if row["run_id"] != context.run_id or row["checkpoint_sha256"] != checkpoint.sha256:
            raise OutputContractError("Prediction run/checkpoint binding mismatch.")
        if row["split"] != "DEV_VALIDATION":
            raise OutputContractError("Only DEV_VALIDATION predictions are permitted.")
        label = row["true_label"]
        if isinstance(label, bool) or label not in (0, 1):
            raise OutputContractError("true_label must be GOOD=0 or BAD=1.")
        if _CLASS_TO_LABEL.get(str(row["original_class"])) != label:
            raise OutputContractError("original_class/true_label drift.")
        relative_path = str(row["relative_path"])
        if relative_path in seen:
            raise OutputContractError("Duplicate validation relative_path.")
        seen.add(relative_path)
        if expected_records.get(relative_path) != (str(row["original_class"]), int(label)):
            raise OutputContractError("Prediction membership differs from verified validation manifest.")
        labels[int(label)] += 1
        p_bad, p_good = float(row["p_BAD"]), float(row["p_GOOD"])
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in (p_bad, p_good)):
            raise OutputContractError("Invalid prediction probabilities.")
        if abs(p_bad + p_good - 1.0) >= 1e-6:
            raise OutputContractError("p_BAD + p_GOOD must equal 1.")
        if int(row["predicted_label_at_0_5"]) != int(p_bad >= 0.5):
            raise OutputContractError("Fixed-threshold prediction drift.")
    expected_labels = {
        0: int(context.manifest.counts["DEV_VALIDATION"]["GOOD"]),
        1: int(context.manifest.counts["DEV_VALIDATION"]["BAD"]),
    }
    if labels != expected_labels or seen != set(expected_records):
        raise OutputContractError(f"Validation label counts mismatch: {labels}")
    return materialized


def validate_phase_metric_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    materialized = tuple(dict(row) for row in rows)
    if not materialized:
        raise OutputContractError("phase_metrics must contain executed epochs.")
    expected_global = 1
    last_epoch = {"P1": 0, "P2": 0}
    seen_p2 = False
    for row in materialized:
        missing = set(PHASE_METRIC_COLUMNS) - set(row)
        if missing:
            raise OutputContractError(f"phase_metrics fields missing: {sorted(missing)}")
        phase = row["phase"]
        if phase not in {"P1", "P2"} or (phase == "P1" and seen_p2):
            raise OutputContractError("Invalid P1/P2 ordering.")
        seen_p2 = seen_p2 or phase == "P2"
        last_epoch[phase] += 1
        if int(row["epoch_in_phase"]) != last_epoch[phase] or int(row["global_epoch"]) != expected_global:
            raise OutputContractError("Invalid phase/global epoch sequence.")
        for key in set(PHASE_METRIC_COLUMNS) - {"phase", "collapse_flag"}:
            if not math.isfinite(float(row[key])):
                raise OutputContractError(f"Non-finite phase metric: {key}")
        expected_global += 1
    if not seen_p2:
        raise OutputContractError("Both P1 and P2 evidence are required.")
    return materialized


def validate_threshold_sweep(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    materialized = tuple(dict(row) for row in rows)
    expected = [index / 100.0 for index in range(1, 100)]
    if [float(row.get("threshold", -1)) for row in materialized] != expected:
        raise OutputContractError("Threshold sweep must be exact ordered points 0.01..0.99.")
    for row in materialized:
        missing = THRESHOLD_FIELDS - set(row)
        if missing:
            raise OutputContractError(f"Threshold fields missing: {sorted(missing)}")
        if any(not math.isfinite(float(row[key])) for key in THRESHOLD_FIELDS):
            raise OutputContractError("Threshold row contains non-finite values.")
    return materialized


def validate_metrics_against_predictions(
    predictions: Iterable[Mapping[str, Any]], metrics: Mapping[str, Any]
) -> None:
    rows = list(predictions)
    recomputed = calculate_metrics(
        [int(row["true_label"]) for row in rows],
        [float(row["p_BAD"]) for row in rows], threshold=0.5,
    )
    for key in ("n", "accuracy", "precision_bad", "recall_bad", "f1_bad", "macro_f1",
                "roc_auc", "pr_auc", "brier", "fn_bad", "fp_good", "fpr_good", "confusion_matrix"):
        if key not in metrics or metrics[key] != recomputed[key]:
            raise OutputContractError(f"Metrics/predictions mismatch: {key}")


def validate_global_best_metadata(
    metadata: Mapping[str, Any], context: VerifiedRunContext,
    best: VerifiedCheckpoint, last: VerifiedCheckpoint,
    phase_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    _assert_context_checkpoint(context, best)
    expected_fingerprint = context_fingerprint(
        context.config.sha256, context.config.protocol_sha256,
        context.manifest.sha256, context.approval.sha256,
    )
    if (
        last.kind != "LAST"
        or last.run_id != context.run_id
        or last.context_fingerprint != expected_fingerprint
        or last.path.resolve() != best.path.with_name("last_model.keras").resolve()
        or not last.path.is_file()
    ):
        raise OutputContractError("Last checkpoint binding is invalid.")
    if sha256_file(last.path) != last.sha256:
        raise OutputContractError("Last checkpoint SHA-256 mismatch.")
    rows = tuple(phase_rows)
    if not rows:
        raise OutputContractError("Global-best metadata requires phase metrics.")
    minimum = min(rows, key=lambda row: float(row["val_loss"]))
    expected = {
        "run_id": context.run_id,
        "best_phase": minimum["phase"],
        "best_epoch_in_phase": int(minimum["epoch_in_phase"]),
        "best_global_epoch": int(minimum["global_epoch"]),
        "best_val_loss": float(minimum["val_loss"]),
        "best_path": str(best.path),
        "best_sha256": best.sha256,
        "last_path": str(last.path),
        "last_sha256": last.sha256,
        "checkpoint_selection_metric": "val_loss",
        "evaluation_model_source": "GLOBAL_BEST_CHECKPOINT",
    }
    if dict(metadata) != expected:
        raise OutputContractError("Global-best metadata/phase/checkpoint mismatch.")
    return expected


def validate_best_vs_last(
    payload: Mapping[str, Any], best: VerifiedCheckpoint, last: VerifiedCheckpoint
) -> dict[str, Any]:
    expected = {
        "status": "HASH_COMPARISON_ONLY",
        "best_sha256": best.sha256,
        "last_sha256": last.sha256,
        "same_bytes": best.sha256 == last.sha256,
    }
    if dict(payload) != expected:
        raise OutputContractError("best_vs_last/checkpoint mismatch.")
    return expected


def validate_per_original_class(
    rows: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]]
) -> tuple[dict[str, Any], ...]:
    materialized = tuple(dict(row) for row in rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for prediction in predictions:
        grouped.setdefault(str(prediction["original_class"]), []).append(dict(prediction))
    if {str(row.get("original_class")) for row in materialized} != set(grouped):
        raise OutputContractError("Per-class rows do not match prediction classes.")
    for row in materialized:
        original_class = str(row["original_class"])
        items = grouped[original_class]
        probabilities = [float(item["p_BAD"]) for item in items]
        correct = sum(int(item["predicted_label_at_0_5"]) == int(item["true_label"]) for item in items)
        expected_bad_recall = (
            sum(int(item["predicted_label_at_0_5"]) == 1 for item in items) / len(items)
            if all(int(item["true_label"]) == 1 for item in items) else None
        )
        expected = {
            "original_class": original_class,
            "total": len(items),
            "correct": correct,
            "incorrect": len(items) - correct,
            "mean_p_BAD": sum(probabilities) / len(probabilities),
            "median_p_BAD": sorted(probabilities)[len(probabilities) // 2]
            if len(probabilities) % 2 else
            sum(sorted(probabilities)[len(probabilities) // 2 - 1:len(probabilities) // 2 + 1]) / 2,
            "bad_recall": expected_bad_recall,
        }
        if row != expected:
            raise OutputContractError(f"Per-original-class mismatch: {original_class}")
    return materialized


def validate_run_status(
    payload: Mapping[str, Any], context: VerifiedRunContext | None = None,
    checkpoint: VerifiedCheckpoint | None = None,
) -> None:
    required = {
        "status", "stage", "backbone", "run_id", "source_commit", "dirty_tree",
        "frozen_protocol_sha256", "config_sha256", "manifest_sha256",
        "checkpoint_sha256", "training_completed", "holdout_access", "artifact_paths",
        "exception_approvals", "scientific_test_status", "gate_details",
    }
    if not required.issubset(payload):
        raise OutputContractError(f"run_status fields missing: {sorted(required - set(payload))}")
    if payload["status"] not in RUN_STATUSES:
        raise OutputContractError("run_status.status must be PASS, FAIL, or INCOMPLETE.")
    if any(value not in GATE_STATUSES for value in dict(payload["gate_details"]).values()):
        raise OutputContractError("run_status gate detail is invalid.")
    if context is not None or checkpoint is not None:
        if context is None or checkpoint is None:
            raise TypeError("run_status binding validation requires both context and checkpoint.")
        _assert_context_checkpoint(context, checkpoint)
        expected = {
            "run_id": context.run_id,
            "source_commit": context.approval.source_commit,
            "frozen_protocol_sha256": context.config.protocol_sha256,
            "config_sha256": context.config.sha256,
            "manifest_sha256": context.manifest.sha256,
            "checkpoint_sha256": checkpoint.sha256,
        }
        if any(payload[key] != value for key, value in expected.items()):
            raise OutputContractError("run_status/context/checkpoint binding mismatch.")
        if set(dict(payload["artifact_paths"])) != REQUIRED_RUN_ARTIFACTS:
            raise OutputContractError("run_status artifact inventory declaration mismatch.")


def validate_artifact_inventory(run_directory: Path) -> None:
    present = {path.name for path in run_directory.iterdir() if path.is_file()}
    missing = REQUIRED_RUN_ARTIFACTS - present
    if missing:
        raise OutputContractError(f"Required run artifacts missing: {sorted(missing)}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Iterable[str]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    fieldnames = list(columns)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
