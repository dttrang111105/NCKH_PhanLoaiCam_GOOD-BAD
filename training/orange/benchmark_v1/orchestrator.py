"""Fail-closed Binary V1 preflight and isolated-run orchestration."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .approval import build_run_context, revalidate_run_context, verify_approval
from .config import verify_frozen_config
from .context import (
    IsolatedRunDirectory,
    VerifiedRunContext,
    _DIRECTORY_SEAL,
    context_fingerprint,
)
from .manifest import verify_manifest


@dataclass(frozen=True)
class PreflightResult:
    status: str
    reason: str
    context: VerifiedRunContext | None = None
    acceptance_status: str = "UNVERIFIED"
    gate_details: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunOutcome:
    run_directory: IsolatedRunDirectory
    training_result: Any
    validation_evidence: Any


def preflight(
    *, config_path: str | Path, protocol_path: str | Path,
    approval_path: str | Path | None = None,
    trusted_approval_sha256: str | None = None,
) -> PreflightResult:
    """Validate all gates without building a model, reading images, or training."""
    try:
        config = verify_frozen_config(config_path, protocol_path)
        manifest = verify_manifest(config)
        if approval_path is None or trusted_approval_sha256 is None:
            return PreflightResult(
                "BLOCKED", "Approved record and independently trusted hash are required.",
                gate_details={"A": "UNVERIFIED", "B": "BLOCKED", "C": "NOT_RUN", "D": "NOT_RUN"},
            )
        approval = verify_approval(
            approval_path,
            expected_approval_sha256=trusted_approval_sha256,
            config=config,
            manifest=manifest,
        )
        return PreflightResult(
            "VERIFIED_RUN_CONTEXT",
            "Config, protocol, manifest, approval, and Git bindings verified; acceptance is not implied.",
            build_run_context(config=config, manifest=manifest, approval=approval),
            acceptance_status="UNVERIFIED",
            gate_details={"A": "UNVERIFIED", "B": "PASS", "C": "NOT_RUN", "D": "NOT_RUN"},
        )
    except (OSError, TypeError, ValueError, PermissionError) as exc:
        return PreflightResult(
            "BLOCKED", str(exc), acceptance_status="UNVERIFIED",
            gate_details={"A": "UNVERIFIED", "B": "BLOCKED", "C": "NOT_RUN", "D": "NOT_RUN"},
        )


def create_isolated_run_directory(context: VerifiedRunContext) -> IsolatedRunDirectory:
    if not isinstance(context, VerifiedRunContext):
        raise TypeError("A VerifiedRunContext is required.")
    output_root = (context.config.repo_root / str(context.config.values["project"]["output_root"])).resolve()
    expected_root = (context.config.repo_root / "outputs_v1").resolve()
    if output_root != expected_root:
        raise ValueError("Output root is not the isolated outputs_v1 directory.")
    legacy_root = (context.config.repo_root / "training" / "outputs").resolve()
    if output_root == legacy_root or legacy_root in output_root.parents:
        raise ValueError("Refusing legacy artifact path.")
    output_root.mkdir(parents=True, exist_ok=True)
    run_path = output_root / context.run_id
    run_path.mkdir(exist_ok=False)
    fingerprint = context_fingerprint(
        context.config.sha256,
        context.config.protocol_sha256,
        context.manifest.sha256,
        context.approval.sha256,
    )
    marker = {
        "schema_version": "NCKH_BASELINE_V1_RUN_DIRECTORY",
        "run_id": context.run_id,
        "stage": context.stage,
        "context_fingerprint": fingerprint,
    }
    (run_path / "context.json").write_text(
        json.dumps(marker, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return IsolatedRunDirectory(run_path, context.run_id, fingerprint, _DIRECTORY_SEAL)


def execute_approved_run(context: VerifiedRunContext) -> RunOutcome:
    """Public execution entrypoint for the standard approved workflow.

    It prevents accidental workflow bypass; it is not a sandbox against a caller
    able to alter/import arbitrary Python internals.
    """
    if not isinstance(context, VerifiedRunContext):
        raise TypeError("execute_approved_run requires VerifiedRunContext.")
    from .data import build_train_dataset, build_validation_dataset
    from .evaluation import evaluate_validation
    from .output import (
        PHASE_METRIC_COLUMNS, PREDICTION_COLUMNS, THRESHOLD_COLUMNS,
        REQUIRED_RUN_ARTIFACTS, data_integrity_payload, fixed_metrics_payload,
        validate_artifact_inventory, validate_best_vs_last,
        validate_global_best_metadata, validate_per_original_class, validate_run_status,
        validate_metrics_against_predictions, validate_phase_metric_rows,
        validate_prediction_rows, validate_threshold_sweep, write_csv, write_json,
    )
    from .training import _fit_two_phase, build_model, compute_class_weights

    initial_integrity = revalidate_run_context(context)
    run_directory = create_isolated_run_directory(context)
    run_path = run_directory.path
    with (run_path / "config_snapshot.yaml").open("xb") as stream:
        stream.write(context.config.snapshot_bytes)
    (run_path / "dataset_manifest_sha256.txt").write_text(
        context.manifest.sha256 + "\n", encoding="ascii", errors="strict"
    )
    (run_path / "protocol_sha256.txt").write_text(
        context.config.protocol_sha256 + "\n", encoding="ascii", errors="strict"
    )
    train = build_train_dataset(context.config, context.manifest.train)
    validation = build_validation_dataset(context.config, context.manifest.validation)
    weights = compute_class_weights(context.manifest.train)
    model, backbone = build_model(context.config)
    pre_training_integrity = revalidate_run_context(context)
    started = time.perf_counter()
    training_result = _fit_two_phase(
        context=context, run_directory=run_directory, model=model, backbone=backbone,
        train=train, validation=validation, class_weights=weights,
    )
    training_seconds = time.perf_counter() - started
    evidence = evaluate_validation(
        context=context, run_directory=run_directory, validation=validation,
        checkpoint=training_result.best_checkpoint,
    )

    predictions = validate_prediction_rows(
        evidence.predictions, context, training_result.best_checkpoint
    )
    phase_rows = validate_phase_metric_rows(training_result.phase_metrics)
    threshold_rows = validate_threshold_sweep(evidence.threshold_rows)
    validate_metrics_against_predictions(predictions, evidence.fixed_metrics)
    metrics_payload = fixed_metrics_payload(
        context, training_result.best_checkpoint, evidence.fixed_metrics
    )
    class_rows = validate_per_original_class(evidence.per_original_class, predictions)
    global_best_metadata = validate_global_best_metadata(
        {
            "run_id": context.run_id,
            "best_phase": training_result.global_best.best_phase,
            "best_epoch_in_phase": training_result.global_best.best_epoch_in_phase,
            "best_global_epoch": training_result.global_best.best_global_epoch,
            "best_val_loss": training_result.global_best.best_val_loss,
            "best_path": str(training_result.best_checkpoint.path),
            "best_sha256": training_result.best_checkpoint.sha256,
            "last_path": str(training_result.last_checkpoint.path),
            "last_sha256": training_result.last_checkpoint.sha256,
            "checkpoint_selection_metric": "val_loss",
            "evaluation_model_source": "GLOBAL_BEST_CHECKPOINT",
        },
        context, training_result.best_checkpoint, training_result.last_checkpoint, phase_rows,
    )
    best_vs_last = validate_best_vs_last({
        "status": "HASH_COMPARISON_ONLY",
        "best_sha256": training_result.best_checkpoint.sha256,
        "last_sha256": training_result.last_checkpoint.sha256,
        "same_bytes": training_result.best_checkpoint.sha256 == training_result.last_checkpoint.sha256,
    }, training_result.best_checkpoint, training_result.last_checkpoint)
    final_integrity = revalidate_run_context(context)

    import tensorflow as tf
    environment = {
        "status": "EXECUTED",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "visible_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "seed": context.config.values["project"]["seed"],
        "mixed_precision": False,
        "dtype": "float32",
        "framework": "TensorFlow",
        "reproducibility_scope": (
            "Seeded deterministic input ordering and seeded Keras augmentation; "
            "bitwise identity across devices/framework builds is not claimed."
        ),
    }
    write_json(run_path / "environment.json", environment)
    write_json(run_path / "source_provenance.json", {
        "run_id": context.run_id,
        "source_commit": context.approval.source_commit,
        "config_sha256": context.config.sha256,
        "protocol_sha256": context.config.protocol_sha256,
        "manifest_sha256": context.manifest.sha256,
        "approval_sha256": context.approval.sha256,
        "context_fingerprint": run_directory.context_fingerprint,
        "provenance_sha256": context.manifest.provenance_sha256,
        "created_at_utc": final_integrity.checked_at_utc,
        "integrity_checks": {
            "initial": initial_integrity.checked_at_utc,
            "pre_training": pre_training_integrity.checked_at_utc,
            "pre_finalize": final_integrity.checked_at_utc,
        },
    })
    write_json(run_path / "data_integrity.json", data_integrity_payload(context, {
        "train_file_verification": "PASS", "validation_file_verification": "PASS",
    }))

    history_rows: list[dict[str, Any]] = []
    global_epoch = 0
    for phase, history in (("P1", training_result.phase1_history), ("P2", training_result.phase2_history)):
        length = len(next(iter(history.values()), []))
        for index in range(length):
            global_epoch += 1
            row = {"phase": phase, "epoch_in_phase": index + 1, "global_epoch": global_epoch}
            row.update({key: values[index] for key, values in history.items()})
            history_rows.append(row)
    history_columns = list(dict.fromkeys(key for row in history_rows for key in row))
    write_csv(run_path / "training_history.csv", history_rows, history_columns)
    write_csv(run_path / "phase_metrics.csv", phase_rows, PHASE_METRIC_COLUMNS)

    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (run_path / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    layer_rows = [{
        "name": layer.name, "class_name": layer.__class__.__name__,
        "trainable": bool(layer.trainable), "parameter_count": int(layer.count_params()),
    } for layer in model.layers]
    write_csv(run_path / "layer_inventory.csv", layer_rows,
              ("name", "class_name", "trainable", "parameter_count"))
    write_json(run_path / "global_best_metadata.json", global_best_metadata)
    write_csv(run_path / "validation_predictions.csv", predictions, PREDICTION_COLUMNS)
    write_json(run_path / "validation_metrics_fixed_0_5.json", metrics_payload)
    class_columns = (
        "original_class", "total", "correct", "incorrect", "mean_p_BAD",
        "median_p_BAD", "bad_recall",
    )
    write_csv(run_path / "validation_original_class_metrics.csv", class_rows, class_columns)
    write_csv(run_path / "threshold_sweep.csv", threshold_rows, THRESHOLD_COLUMNS)
    best_diagnostic = max(threshold_rows, key=lambda row: (
        row["f1_bad"], row["recall_bad"], row["accuracy"], -row["fn_bad"]
    ))
    write_json(run_path / "validation_threshold_diagnostic.json", {
        "status": "SECONDARY_DEV_VALIDATION_ONLY", "primary_threshold_unchanged": 0.5,
        "diagnostic": best_diagnostic,
    })
    write_json(run_path / "best_vs_last.json", best_vs_last)
    write_json(run_path / "training_resource_summary.json", {
        "training_seconds": training_seconds,
        "visible_gpus": environment["visible_gpus"],
        "platform": environment["platform"],
        "latency_status": "NOT_RUN",
    })
    missing_before_status = (REQUIRED_RUN_ARTIFACTS - {"run_status.json"}) - {
        path.name for path in run_path.iterdir() if path.is_file()
    }
    if missing_before_status:
        raise RuntimeError(f"Cannot finalize run; artifacts missing: {sorted(missing_before_status)}")
    run_status = {
        # Training/evaluation completion alone is not sufficient for PASS.  This
        # pipeline currently has no independently instrumented holdout-access
        # counter, so it must remain UNVERIFIED instead of inventing zeroes.
        "status": "INCOMPLETE",
        "stage": context.stage,
        "backbone": "MobileNetV3-Small",
        "run_id": context.run_id,
        "source_commit": context.approval.source_commit,
        "dirty_tree": final_integrity.dirty_tree,
        "frozen_protocol_sha256": context.config.protocol_sha256,
        "config_sha256": context.config.sha256,
        "manifest_sha256": context.manifest.sha256,
        "checkpoint_sha256": training_result.best_checkpoint.sha256,
        "training_completed": True,
        "holdout_access": None,
        "holdout_evidence_status": "NOT_INSTRUMENTED",
        "status_reason": "Execution complete; independent holdout-access evidence is unavailable.",
        "artifact_paths": {name: name for name in sorted(REQUIRED_RUN_ARTIFACTS)},
        "exception_approvals": [],
        "scientific_test_status": "NOT_AVAILABLE",
        "gate_details": {
            "SOURCE_IDENTITY": "UNVERIFIED",
            "FROZEN_DATA": "PASS",
            "PREPROCESSING_MODEL": "UNVERIFIED",
            "METRICS_OUTPUT": "UNVERIFIED",
            "CONTROLLED_PILOT": "UNVERIFIED" if context.stage == "PILOT" else "NOT_RUN",
            "FULL_BASELINE": "UNVERIFIED" if context.stage == "FULL" else "NOT_RUN",
        },
    }
    validate_run_status(run_status, context, training_result.best_checkpoint)
    write_json(run_path / "run_status.json", run_status)
    validate_artifact_inventory(run_path)
    return RunOutcome(run_directory, training_result, evidence)
