"""Synthetic output and cross-artifact tests; no project artifacts are created."""

from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tests"))

from benchmark_v1_fixtures import verified_fixture  # noqa: E402
from orange.benchmark_v1.context import (  # noqa: E402
    VerifiedCheckpoint, _CHECKPOINT_SEAL, context_fingerprint,
)
from orange.benchmark_v1.metrics import calculate_metrics, threshold_sweep  # noqa: E402
from orange.benchmark_v1.output import (  # noqa: E402
    REQUIRED_RUN_ARTIFACTS, OutputContractError, data_integrity_payload, fixed_metrics_payload,
    incomplete_run_status, validate_metrics_against_predictions,
    validate_best_vs_last, validate_global_best_metadata,
    validate_per_original_class, validate_phase_metric_rows,
    validate_prediction_rows, validate_run_status, validate_threshold_sweep,
)


def _checkpoint(context, kind="GLOBAL_BEST", sha=None):
    name = "best_model.keras" if kind == "GLOBAL_BEST" else "last_model.keras"
    path = context.config.repo_root / "outputs_v1" / context.run_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic checkpoint bytes")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint = context_fingerprint(
        context.config.sha256, context.config.protocol_sha256,
        context.manifest.sha256, context.approval.sha256,
    )
    return VerifiedCheckpoint(
        path, actual_sha if sha is None else sha, kind, context.run_id,
        fingerprint, _CHECKPOINT_SEAL,
    )


def _predictions(context, checkpoint):
    probabilities = [0.2, 0.7, 0.9]
    rows = []
    for record, probability in zip(context.manifest.validation.records, probabilities):
        rows.append({
            "run_id": context.run_id, "relative_path": record.relative_path,
            "original_class": record.original_class, "split": "DEV_VALIDATION",
            "true_label": record.label_id, "p_BAD": probability,
            "p_GOOD": 1.0 - probability,
            "predicted_label_at_0_5": int(probability >= 0.5),
            "checkpoint_sha256": checkpoint.sha256,
        })
    return rows


def _last_checkpoint(context):
    path = context.config.repo_root / "outputs_v1" / context.run_id / "last_model.keras"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic last checkpoint bytes")
    fingerprint = context_fingerprint(
        context.config.sha256, context.config.protocol_sha256,
        context.manifest.sha256, context.approval.sha256,
    )
    return VerifiedCheckpoint(
        path, hashlib.sha256(path.read_bytes()).hexdigest(), "LAST",
        context.run_id, fingerprint, _CHECKPOINT_SEAL,
    )


def _phase(phase, epoch, global_epoch):
    return {
        "phase": phase, "epoch_in_phase": epoch, "global_epoch": global_epoch,
        "train_loss": 1.0, "train_accuracy": 0.5, "val_loss": 1.0,
        "val_accuracy": 0.5, "roc_auc": 0.5, "pr_auc": 0.5,
        "pbad_mean": 0.5, "pbad_std": 0.1, "good_pbad_median": 0.4,
        "bad_pbad_median": 0.6, "median_separation": 0.2,
        "fn_bad_0_5": 0, "fp_good_0_5": 0, "fpr_good_0_5": 0.0,
        "collapse_flag": "OK",
    }


class OutputContractTests(unittest.TestCase):
    def test_nonexecuted_status_does_not_forge_holdout_zero(self):
        status = incomplete_run_status(context=None, reason="synthetic", gate_status="BLOCKED")
        self.assertEqual(status["status"], "INCOMPLETE")
        self.assertEqual(status["gate_status"], "BLOCKED")
        self.assertFalse(status["training_completed"])
        self.assertIsNone(status["holdout_access"])
        self.assertEqual(status["holdout_evidence_status"], "NOT_RUN")

    def test_primary_payload_protects_threshold_and_required_fields(self):
        with verified_fixture() as fixture:
            checkpoint = _checkpoint(fixture.context)
            rows = _predictions(fixture.context, checkpoint)
            metrics = dict(calculate_metrics(
                [row["true_label"] for row in rows], [row["p_BAD"] for row in rows]
            ))
            metrics["val_loss"] = 0.25
            payload = fixed_metrics_payload(fixture.context, checkpoint, metrics)
            self.assertEqual(payload["threshold"], 0.5)
            missing = dict(metrics)
            missing.pop("brier")
            with self.assertRaises(OutputContractError):
                fixed_metrics_payload(fixture.context, checkpoint, missing)
            with self.assertRaises(OutputContractError):
                fixed_metrics_payload(fixture.context, checkpoint, dict(metrics, threshold=0.4))

    def test_predictions_are_bound_to_manifest_run_and_checkpoint(self):
        with verified_fixture() as fixture:
            checkpoint = _checkpoint(fixture.context)
            rows = _predictions(fixture.context, checkpoint)
            validate_prediction_rows(rows, fixture.context, checkpoint)
            for field, forged in (
                ("run_id", "forged"), ("checkpoint_sha256", "e" * 64),
                ("relative_path", "fresh_orange/not_in_manifest.jpg"),
            ):
                changed = copy.deepcopy(rows)
                changed[0][field] = forged
                with self.assertRaises(OutputContractError):
                    validate_prediction_rows(changed, fixture.context, checkpoint)

    def test_forged_checkpoint_sha_is_rejected(self):
        with verified_fixture() as fixture:
            checkpoint = _checkpoint(fixture.context, sha="e" * 64)
            with self.assertRaisesRegex(OutputContractError, "SHA-256 mismatch"):
                validate_prediction_rows([], fixture.context, checkpoint)

    def test_label_and_fixed_threshold_drift_are_rejected(self):
        with verified_fixture() as fixture:
            checkpoint = _checkpoint(fixture.context)
            rows = _predictions(fixture.context, checkpoint)
            changed = copy.deepcopy(rows)
            changed[0]["true_label"] = 1
            with self.assertRaises(OutputContractError):
                validate_prediction_rows(changed, fixture.context, checkpoint)
            changed = copy.deepcopy(rows)
            changed[0]["predicted_label_at_0_5"] = 1
            with self.assertRaises(OutputContractError):
                validate_prediction_rows(changed, fixture.context, checkpoint)

    def test_metrics_prediction_mismatch_is_rejected(self):
        with verified_fixture() as fixture:
            checkpoint = _checkpoint(fixture.context)
            rows = _predictions(fixture.context, checkpoint)
            metrics = dict(calculate_metrics(
                [row["true_label"] for row in rows], [row["p_BAD"] for row in rows]
            ))
            metrics["accuracy"] = 0.0
            with self.assertRaisesRegex(OutputContractError, "accuracy"):
                validate_metrics_against_predictions(rows, metrics)

    def test_threshold_schema_requires_all_99_complete_points(self):
        rows = threshold_sweep([0, 1], [0.2, 0.8])
        validate_threshold_sweep(rows)
        with self.assertRaises(OutputContractError):
            validate_threshold_sweep(rows[:-1])
        incomplete = [dict(row) for row in rows]
        incomplete[0].pop("accuracy")
        with self.assertRaises(OutputContractError):
            validate_threshold_sweep(incomplete)

    def test_phase_metrics_require_both_ordered_phases(self):
        validate_phase_metric_rows([_phase("P1", 1, 1), _phase("P2", 1, 2)])
        with self.assertRaises(OutputContractError):
            validate_phase_metric_rows([_phase("P1", 1, 1)])

    def test_data_integrity_requires_explicit_evidence(self):
        with verified_fixture() as fixture:
            payload = data_integrity_payload(fixture.context, {
                "train_file_verification": "NOT_RUN",
                "validation_file_verification": "NOT_RUN",
            })
            self.assertIsNone(payload["holdout_access"])
            with self.assertRaises(OutputContractError):
                data_integrity_payload(fixture.context, {})

    def test_global_best_metadata_and_best_vs_last_are_cross_checked(self):
        with verified_fixture() as fixture:
            best = _checkpoint(fixture.context)
            last = _last_checkpoint(fixture.context)
            rows = [_phase("P1", 1, 1), _phase("P2", 1, 2)]
            rows[0]["val_loss"] = 0.4
            rows[1]["val_loss"] = 0.2
            metadata = {
                "run_id": fixture.context.run_id,
                "best_phase": "P2", "best_epoch_in_phase": 1,
                "best_global_epoch": 2, "best_val_loss": 0.2,
                "best_path": str(best.path), "best_sha256": best.sha256,
                "last_path": str(last.path), "last_sha256": last.sha256,
                "checkpoint_selection_metric": "val_loss",
                "evaluation_model_source": "GLOBAL_BEST_CHECKPOINT",
            }
            validate_global_best_metadata(metadata, fixture.context, best, last, rows)
            comparison = {
                "status": "HASH_COMPARISON_ONLY", "best_sha256": best.sha256,
                "last_sha256": last.sha256, "same_bytes": False,
            }
            validate_best_vs_last(comparison, best, last)
            with self.assertRaises(OutputContractError):
                validate_global_best_metadata(dict(metadata, best_phase="P1"), fixture.context, best, last, rows)

    def test_per_original_class_bad_recall_is_validated(self):
        with verified_fixture() as fixture:
            checkpoint = _checkpoint(fixture.context)
            predictions = _predictions(fixture.context, checkpoint)
            rows = []
            for original_class in sorted({row["original_class"] for row in predictions}):
                items = [row for row in predictions if row["original_class"] == original_class]
                probabilities = [row["p_BAD"] for row in items]
                correct = sum(row["true_label"] == row["predicted_label_at_0_5"] for row in items)
                rows.append({
                    "original_class": original_class, "total": len(items),
                    "correct": correct, "incorrect": len(items) - correct,
                    "mean_p_BAD": sum(probabilities) / len(probabilities),
                    "median_p_BAD": probabilities[0],
                    "bad_recall": 1.0 if items[0]["true_label"] == 1 else None,
                })
            validate_per_original_class(rows, predictions)
            forged = copy.deepcopy(rows)
            next(row for row in forged if row["bad_recall"] is not None)["bad_recall"] = 0.0
            with self.assertRaises(OutputContractError):
                validate_per_original_class(forged, predictions)

    def test_run_status_uses_contract_status_and_gate_details(self):
        payload = {
            "status": "INCOMPLETE", "stage": "PREFLIGHT", "backbone": "MobileNetV3-Small",
            "run_id": None, "source_commit": None, "dirty_tree": None,
            "frozen_protocol_sha256": None, "config_sha256": None,
            "manifest_sha256": None, "checkpoint_sha256": None,
            "training_completed": False, "holdout_access": None,
            "artifact_paths": {}, "exception_approvals": [],
            "scientific_test_status": "NOT_AVAILABLE",
            "gate_details": {"SOURCE_IDENTITY": "UNVERIFIED"},
        }
        validate_run_status(payload)
        with self.assertRaises(OutputContractError):
            validate_run_status(dict(payload, status="UNVERIFIED"))

    def test_run_status_is_bound_to_context_checkpoint_and_inventory(self):
        with verified_fixture() as fixture:
            checkpoint = _checkpoint(fixture.context)
            payload = {
                "status": "INCOMPLETE", "stage": fixture.context.stage,
                "backbone": "MobileNetV3-Small", "run_id": fixture.context.run_id,
                "source_commit": fixture.context.approval.source_commit,
                "dirty_tree": False,
                "frozen_protocol_sha256": fixture.context.config.protocol_sha256,
                "config_sha256": fixture.context.config.sha256,
                "manifest_sha256": fixture.context.manifest.sha256,
                "checkpoint_sha256": checkpoint.sha256,
                "training_completed": True, "holdout_access": None,
                "artifact_paths": {name: name for name in REQUIRED_RUN_ARTIFACTS},
                "exception_approvals": [], "scientific_test_status": "NOT_AVAILABLE",
                "gate_details": {"METRICS_OUTPUT": "UNVERIFIED"},
            }
            validate_run_status(payload, fixture.context, checkpoint)
            with self.assertRaisesRegex(OutputContractError, "binding mismatch"):
                validate_run_status(
                    dict(payload, run_id="forged"), fixture.context, checkpoint
                )


if __name__ == "__main__":
    unittest.main()
