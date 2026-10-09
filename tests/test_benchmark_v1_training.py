"""Synthetic training-contract tests; TensorFlow and fit() are not invoked."""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tests"))

from benchmark_v1_fixtures import verified_fixture  # noqa: E402
import orange.benchmark_v1 as public_api  # noqa: E402
from orange.benchmark_v1.context import (  # noqa: E402
    IsolatedRunDirectory, VerifiedClassWeights, VerifiedTrainDataset,
    VerifiedValidationDataset, _DATASET_SEAL, _DIRECTORY_SEAL, _WEIGHT_SEAL,
    context_fingerprint,
)
from orange.benchmark_v1.training import (  # noqa: E402
    GlobalBestState, TrainingContractError, _validate_runtime_bindings,
    _fit_two_phase, compute_class_weights, select_global_best,
)


class TrainingContractTests(unittest.TestCase):
    def test_supported_public_execution_api_excludes_fit_helper(self):
        self.assertIn("execute_approved_run", public_api.__all__)
        self.assertNotIn("_fit_two_phase", public_api.__all__)

    def test_global_best_spans_both_phases(self):
        state = select_global_best([
            {"phase": "P1", "epoch_in_phase": 1, "global_epoch": 1, "val_loss": 0.40},
            {"phase": "P1", "epoch_in_phase": 2, "global_epoch": 2, "val_loss": 0.30},
            {"phase": "P2", "epoch_in_phase": 1, "global_epoch": 3, "val_loss": 0.35},
            {"phase": "P2", "epoch_in_phase": 2, "global_epoch": 4, "val_loss": 0.20},
        ])
        self.assertEqual(state.best_phase, "P2")
        self.assertEqual(state.best_global_epoch, 4)
        self.assertEqual(state.best_val_loss, 0.20)

    def test_equal_loss_does_not_replace_earlier_best(self):
        state = GlobalBestState()
        self.assertTrue(state.observe(phase="P1", epoch_in_phase=1, global_epoch=1, val_loss=0.2))
        self.assertFalse(state.observe(phase="P2", epoch_in_phase=1, global_epoch=2, val_loss=0.2))
        self.assertEqual(state.best_phase, "P1")

    def test_class_weights_come_from_verified_train_only(self):
        with verified_fixture() as fixture:
            weights = compute_class_weights(fixture.context.manifest.train)
            self.assertEqual(dict(weights.values), {0: 1.5, 1: 0.75})
            self.assertEqual(weights.train_count, 3)

    def test_wrong_class_weights_fail_runtime_binding(self):
        with verified_fixture() as fixture:
            context = fixture.context
            fingerprint = context_fingerprint(
                context.config.sha256, context.config.protocol_sha256,
                context.manifest.sha256, context.approval.sha256,
            )
            directory = IsolatedRunDirectory(
                context.config.repo_root / "outputs_v1" / context.run_id,
                context.run_id, fingerprint, _DIRECTORY_SEAL
            )
            train = VerifiedTrainDataset(
                object(), context.manifest.sha256, 3, 16, "NONE",
                context.config.sha256, 224, _DATASET_SEAL
            )
            validation = VerifiedValidationDataset(
                object(), context.manifest.sha256, 3, 16,
                context.config.sha256, 224, _DATASET_SEAL
            )
            forged = VerifiedClassWeights(
                {0: 1.0, 1: 1.0}, context.manifest.sha256, 3, _WEIGHT_SEAL
            )
            with self.assertRaisesRegex(TrainingContractError, "verified DEV_TRAIN"):
                _validate_runtime_bindings(context, directory, train, validation, forged)

    def test_arbitrary_dataset_is_rejected_before_tensorflow(self):
        with verified_fixture() as fixture:
            with self.assertRaises(TypeError):
                _validate_runtime_bindings(fixture.context, object(), object(), object(), object())

    def test_dataset_preprocessing_binding_drift_is_rejected(self):
        with verified_fixture() as fixture:
            context = fixture.context
            fingerprint = context_fingerprint(
                context.config.sha256, context.config.protocol_sha256,
                context.manifest.sha256, context.approval.sha256,
            )
            directory = IsolatedRunDirectory(
                context.config.repo_root / "outputs_v1" / context.run_id,
                context.run_id, fingerprint, _DIRECTORY_SEAL,
            )
            train = VerifiedTrainDataset(
                object(), context.manifest.sha256, 3, 8, "NONE",
                context.config.sha256, 224, _DATASET_SEAL,
            )
            validation = VerifiedValidationDataset(
                object(), context.manifest.sha256, 3, 16,
                context.config.sha256, 224, _DATASET_SEAL,
            )
            weights = compute_class_weights(context.manifest.train)
            with self.assertRaisesRegex(TrainingContractError, "preprocessing binding"):
                _validate_runtime_bindings(context, directory, train, validation, weights)

    def test_fit_has_no_boolean_approval_escape_hatch(self):
        parameters = inspect.signature(_fit_two_phase).parameters
        self.assertNotIn("approval_recorded", parameters)
        with self.assertRaises(TypeError):
            _fit_two_phase(approval_recorded=True)  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
