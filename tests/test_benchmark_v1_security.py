"""Synthetic negative tests for approval, preflight, and checkpoint gates."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tests"))

from benchmark_v1_fixtures import verified_fixture  # noqa: E402
from orange.benchmark_v1.approval import (  # noqa: E402
    ApprovalVerificationError, revalidate_run_context, verify_approval,
)
from orange.benchmark_v1.context import (  # noqa: E402
    IsolatedRunDirectory, VerifiedCheckpoint, _CHECKPOINT_SEAL, _DIRECTORY_SEAL,
    context_fingerprint,
)
from orange.benchmark_v1.evaluation import (  # noqa: E402
    EvaluationContractError, _validate_best_checkpoint,
)
from orange.benchmark_v1.orchestrator import preflight  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SecurityGateTests(unittest.TestCase):
    def test_clean_context_revalidates(self):
        with verified_fixture() as fixture:
            evidence = revalidate_run_context(fixture.context)
            self.assertFalse(evidence.dirty_tree)
            self.assertEqual(evidence.source_commit, fixture.context.approval.source_commit)

    def test_changed_config_makes_context_stale(self):
        with verified_fixture() as fixture:
            fixture.config_path.write_text(
                fixture.config_path.read_text(encoding="utf-8") + "\n# stale\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ApprovalVerificationError, "config_sha256 mismatch"):
                revalidate_run_context(fixture.context)

    def test_changed_approval_makes_context_stale(self):
        with verified_fixture() as fixture:
            fixture.approval_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ApprovalVerificationError, "approval_sha256 mismatch"):
                revalidate_run_context(fixture.context)

    def test_changed_protocol_manifest_or_provenance_makes_context_stale(self):
        for attribute, expected_name in (
            ("protocol_path", "protocol_sha256"),
            ("manifest_path", "manifest_sha256"),
            ("provenance_path", "provenance_sha256"),
        ):
            with self.subTest(attribute=attribute), verified_fixture() as fixture:
                path = getattr(fixture, attribute)
                path.write_bytes(path.read_bytes() + b"stale")
                with self.assertRaisesRegex(ApprovalVerificationError, expected_name):
                    revalidate_run_context(fixture.context)

    def test_changed_git_head_makes_context_stale(self):
        with verified_fixture() as fixture:
            subprocess.check_call(
                ["git", "commit", "--allow-empty", "-qm", "synthetic head drift"],
                cwd=fixture.repo,
            )
            with self.assertRaisesRegex(ApprovalVerificationError, "Git HEAD changed"):
                revalidate_run_context(fixture.context)

    def test_forged_approval_artifact_fails_trusted_hash(self):
        with verified_fixture() as fixture:
            data = json.loads(fixture.approval_path.read_text(encoding="utf-8"))
            data["approved_by"] = "forged"
            fixture.approval_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ApprovalVerificationError, "SHA-256 mismatch"):
                verify_approval(
                    fixture.approval_path,
                    expected_approval_sha256=fixture.approval_sha256,
                    config=fixture.context.config,
                    manifest=fixture.context.manifest,
                )

    def test_extra_approval_field_is_rejected_even_with_matching_hash(self):
        with verified_fixture() as fixture:
            data = json.loads(fixture.approval_path.read_text(encoding="utf-8"))
            data["approval_recorded"] = True
            fixture.approval_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ApprovalVerificationError, "unexpected fields"):
                verify_approval(
                    fixture.approval_path,
                    expected_approval_sha256=_sha(fixture.approval_path),
                    config=fixture.context.config,
                    manifest=fixture.context.manifest,
                )

    def test_pilot_config_cannot_use_full_approval(self):
        with verified_fixture(profile="pilot_v1") as fixture:
            data = json.loads(fixture.approval_path.read_text(encoding="utf-8"))
            data["stage"] = "FULL"
            fixture.approval_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ApprovalVerificationError, "stage mismatch"):
                verify_approval(
                    fixture.approval_path,
                    expected_approval_sha256=_sha(fixture.approval_path),
                    config=fixture.context.config,
                    manifest=fixture.context.manifest,
                )

    def test_last_checkpoint_cannot_be_evaluated_as_global_best(self):
        with verified_fixture() as fixture:
            context = fixture.context
            fingerprint = context_fingerprint(
                context.config.sha256, context.config.protocol_sha256,
                context.manifest.sha256, context.approval.sha256,
            )
            directory = IsolatedRunDirectory(
                fixture.temp_root / "run", context.run_id, fingerprint, _DIRECTORY_SEAL
            )
            last = VerifiedCheckpoint(
                directory.path / "last_model.keras", "a" * 64, "LAST",
                context.run_id, fingerprint, _CHECKPOINT_SEAL,
            )
            with self.assertRaisesRegex(EvaluationContractError, "GLOBAL_BEST"):
                _validate_best_checkpoint(context, directory, last)

    def test_checkpoint_run_id_forgery_is_rejected(self):
        with verified_fixture() as fixture:
            context = fixture.context
            fingerprint = context_fingerprint(
                context.config.sha256, context.config.protocol_sha256,
                context.manifest.sha256, context.approval.sha256,
            )
            directory = IsolatedRunDirectory(
                fixture.temp_root / "run", context.run_id, fingerprint, _DIRECTORY_SEAL
            )
            forged = VerifiedCheckpoint(
                directory.path / "best_model.keras", "a" * 64, "GLOBAL_BEST",
                "forged-run", fingerprint, _CHECKPOINT_SEAL,
            )
            with self.assertRaisesRegex(EvaluationContractError, "Run ID"):
                _validate_best_checkpoint(context, directory, forged)

    def test_real_draft_preflight_remains_blocked_without_reading_data(self):
        result = preflight(
            config_path=ROOT / "configs" / "baseline_v1.yaml",
            protocol_path=ROOT / "docs" / "PROTOCOL_EXCEPTION_REQUEST.md",
        )
        self.assertEqual(result.status, "BLOCKED")
        self.assertIsNone(result.context)

    def test_verified_context_is_not_acceptance_pass(self):
        with verified_fixture() as fixture:
            result = preflight(
                config_path=fixture.config_path,
                protocol_path=fixture.protocol_path,
                approval_path=fixture.approval_path,
                trusted_approval_sha256=fixture.approval_sha256,
            )
            self.assertEqual(result.status, "VERIFIED_RUN_CONTEXT")
            self.assertEqual(result.acceptance_status, "UNVERIFIED")


if __name__ == "__main__":
    unittest.main()
