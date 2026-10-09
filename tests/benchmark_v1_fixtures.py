"""Synthetic-only verifier fixtures. No project dataset or model is used."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from orange.benchmark_v1 import config as config_module  # noqa: E402
from orange.benchmark_v1 import manifest as manifest_module  # noqa: E402
from orange.benchmark_v1 import training as training_module  # noqa: E402
from orange.benchmark_v1.approval import build_run_context, verify_approval  # noqa: E402
from orange.benchmark_v1.config import verify_frozen_config  # noqa: E402
from orange.benchmark_v1.context import VerifiedRunContext  # noqa: E402
from orange.benchmark_v1.manifest import verify_manifest  # noqa: E402

SYNTHETIC_COUNTS = {
    "DEV_TRAIN": {"GOOD": 1, "BAD": 2, "total": 3},
    "DEV_VALIDATION": {"GOOD": 1, "BAD": 2, "total": 3},
    "DEV_HOLDOUT": {"GOOD": 1, "BAD": 2, "total": 3},
}
SOURCE_SHA = "a" * 64


@dataclass
class SyntheticVerifiedFixture:
    temp_root: Path
    repo: Path
    config_path: Path
    protocol_path: Path
    manifest_path: Path
    provenance_path: Path
    approval_path: Path
    approval_sha256: str
    context: VerifiedRunContext


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def _rows():
    rows = []
    index = 0
    for split in SYNTHETIC_COUNTS:
        for original_class, label, count in (
            ("fresh_orange", 0, 1), ("blackspotted_orange", 1, 1), ("rotten_orange", 1, 1)
        ):
            for _ in range(count):
                index += 1
                rows.append({
                    "relative_path": f"{original_class}/synthetic_{index}.jpg",
                    "original_class": original_class,
                    "binary_label": "GOOD" if label == 0 else "BAD",
                    "label_id": label,
                    "split": split,
                    "source_row_key": f"synthetic-{index}",
                })
    return rows


@contextmanager
def verified_fixture(*, profile: str = "baseline_v1"):
    with tempfile.TemporaryDirectory() as temp, patch.object(
        config_module, "EXPECTED_COUNTS", SYNTHETIC_COUNTS
    ), patch.object(config_module, "SOURCE_MANIFEST_SHA256", SOURCE_SHA), patch.object(
        manifest_module, "EXPECTED_COUNTS", SYNTHETIC_COUNTS
    ), patch.object(manifest_module, "SOURCE_MANIFEST_SHA256", SOURCE_SHA), patch.object(
        training_module, "EXPECTED_COUNTS", SYNTHETIC_COUNTS
    ):
        temp_root = Path(temp)
        repo = temp_root / "repo"
        (repo / "configs").mkdir(parents=True)
        (repo / "shared_data").mkdir()
        (repo / ".gitignore").write_text("outputs_v1/\n", encoding="utf-8")
        protocol = repo / "protocol.md"
        protocol.write_text("SYNTHETIC PROTOCOL — UNIT TEST ONLY\n", encoding="utf-8")
        protocol_sha = _sha(protocol)
        with patch.object(config_module, "FROZEN_PROTOCOL_SHA256", protocol_sha):
            manifest = repo / "shared_data" / "manifest_binary_v1_portable.csv"
            fields = ["relative_path", "original_class", "binary_label", "label_id", "split", "source_row_key"]
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(_rows())
            manifest_sha = _sha(manifest)
            provenance = repo / "shared_data" / "manifest_binary_v1_portable_provenance.json"
            provenance.write_text(json.dumps({
                "verification_status": "PASS",
                "source_manifest_sha256": SOURCE_SHA,
                "portable_manifest_sha256": manifest_sha,
                "transform_script_sha256": "b" * 64,
                "row_counts": SYNTHETIC_COUNTS,
                "label_policy_id": "NCKH_BACKBONE_BENCHMARK_V1",
                "created_at_utc": "2026-10-09T00:00:00Z",
            }), encoding="utf-8")
            pilot = profile == "pilot_v1"
            prefix = "mobilenetv3_small_pilot_v1" if pilot else "mobilenetv3_small_baseline_v1"
            config = {
                "schema_version": "NCKH_BACKBONE_BENCHMARK_V1",
                "profile": profile, "status": "FROZEN", "approval_required": True,
                "project": {"seed": 42, "output_root": "outputs_v1", "run_id_prefix": prefix},
                "data": {
                    "dataset_root_env": "DATASET_ROOT", "image_size": 224, "batch_size": 16,
                    "manifest_path": "shared_data/manifest_binary_v1_portable.csv",
                    "provenance_path": "shared_data/manifest_binary_v1_portable_provenance.json",
                    "expected_source_manifest_sha256": SOURCE_SHA,
                    "expected_portable_manifest_sha256": manifest_sha,
                    "expected_counts": SYNTHETIC_COUNTS,
                    "label_mapping": {"GOOD": 0, "BAD": 1},
                    "excluded_original_classes": ["green_orange"],
                },
                "model": {
                    "constructor": "tf.keras.applications.MobileNetV3Small", "weights": "imagenet",
                    "include_top": False, "include_preprocessing": True, "dropout": 0.20,
                    "output": "sigmoid_p_BAD", "mixed_precision": False,
                },
                "augmentation": {
                    "horizontal_flip": True, "rotation_factor": 0.04, "zoom_factor": 0.10,
                    "translation_factor": 0.05, "contrast_factor": 0.10, "train_only": True,
                },
                "training": {
                    "optimizer": "Adam",
                    "phase1": {"max_epochs": 1 if pilot else 10, "learning_rate": 0.001, "early_stopping_patience": 3},
                    "phase2": {"max_epochs": 1 if pilot else 20, "learning_rate": 0.0001, "early_stopping_patience": 5},
                    "checkpoint_monitor": "val_loss", "checkpoint_mode": "min",
                    "restore_best_weights": False, "reduce_lr_on_plateau": False,
                    "full_backbone_unfreeze_phase2": True, "freeze_batch_normalization_phase2": True,
                },
                "evaluation": {
                    "primary_threshold": 0.5, "threshold_min": 0.01,
                    "threshold_max": 0.99, "threshold_step": 0.01,
                    "scientific_status": "LEGACY_DEV_INTERNAL_NOT_INDEPENDENT",
                    "holdout_access": "FORBIDDEN",
                },
                "contract": {
                    "reference_commit": config_module.REFERENCE_COMMIT,
                    "declared_frozen_protocol_sha256": protocol_sha,
                    "verification_status": "VERIFIED",
                },
            }
            config_path = repo / "configs" / f"{profile}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            _run("git", "init", "-q", cwd=repo)
            _run("git", "config", "user.email", "synthetic@example.invalid", cwd=repo)
            _run("git", "config", "user.name", "Synthetic Fixture", cwd=repo)
            _run("git", "add", ".", cwd=repo)
            _run("git", "commit", "-qm", "synthetic fixture", cwd=repo)
            source_commit = _run("git", "rev-parse", "HEAD", cwd=repo)
            verified_config = verify_frozen_config(config_path, protocol)
            verified_manifest = verify_manifest(verified_config)
            approval = temp_root / "approval.json"
            approval.write_text(json.dumps({
                "schema_version": "NCKH_BASELINE_V1_APPROVAL", "status": "APPROVED",
                "stage": "PILOT" if pilot else "FULL",
                "run_id": prefix + "_synthetic", "approval_id": "SYNTHETIC-ONLY",
                "approved_by": "synthetic-fixture", "approved_at_utc": "2026-10-09T00:00:00Z",
                "config_sha256": verified_config.sha256,
                "protocol_sha256": verified_config.protocol_sha256,
                "manifest_sha256": verified_manifest.sha256,
                "source_manifest_sha256": verified_manifest.source_manifest_sha256,
                "source_commit": source_commit,
            }), encoding="utf-8")
            approval_sha = _sha(approval)
            verified_approval = verify_approval(
                approval, expected_approval_sha256=approval_sha,
                config=verified_config, manifest=verified_manifest,
            )
            context = build_run_context(
                config=verified_config, manifest=verified_manifest, approval=verified_approval
            )
            yield SyntheticVerifiedFixture(
                temp_root, repo, config_path, protocol, manifest, provenance,
                approval, approval_sha, context,
            )
