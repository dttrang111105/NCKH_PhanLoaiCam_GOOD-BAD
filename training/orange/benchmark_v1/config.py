"""Strict draft validation and frozen-config verification for Binary V1."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from .context import VerifiedConfig, _CONFIG_SEAL, _deep_freeze

SCHEMA_VERSION = "NCKH_BACKBONE_BENCHMARK_V1"
REFERENCE_COMMIT = "addaee99f8ba410ac7efd1e6abce34c042561c75"
FROZEN_PROTOCOL_SHA256 = "6cedb8cfed97c07fc19aab4e44d3148b0b931b5bc7ac034505ec3428138b88d0"
SOURCE_MANIFEST_SHA256 = "f7bd8fd28c436e7e23f70e2ad5df27318d28ce528572e92ba7f2cea32b8dfa68"
EXPECTED_COUNTS = {
    "DEV_TRAIN": {"GOOD": 1963, "BAD": 5888, "total": 7851},
    "DEV_VALIDATION": {"GOOD": 517, "BAD": 1542, "total": 2059},
    "DEV_HOLDOUT": {"GOOD": 520, "BAD": 1570, "total": 2090},
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ContractConfigError(ValueError):
    """Raised when a config is malformed or drifts from Binary V1."""


class PreflightBlockedError(ContractConfigError):
    """Raised when structure is valid but frozen evidence is unavailable."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise PreflightBlockedError(f"BLOCKED: config file missing: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ContractConfigError("Benchmark config must be a YAML object.")
    return data


def load_and_validate_config(path: str | Path) -> dict[str, Any]:
    data = load_config(path)
    validate_config(data)
    return data


def _expect(actual: Any, expected: Any, name: str, drift: list[str]) -> None:
    if actual != expected:
        drift.append(name)


def _expect_exact_keys(value: Any, expected: set[str], name: str, drift: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        drift.append(f"{name}.type")
        return {}
    if set(value) != expected:
        drift.append(f"{name}.schema")
    return value


def validate_config(config: dict[str, Any]) -> None:
    drift: list[str] = []
    _expect_exact_keys(config, {
        "schema_version", "profile", "status", "approval_required", "project", "data",
        "model", "augmentation", "training", "evaluation", "contract",
    }, "root", drift)
    project = _expect_exact_keys(config.get("project"), {
        "seed", "output_root", "run_id_prefix",
    }, "project", drift)
    data = _expect_exact_keys(config.get("data"), {
        "dataset_root_env", "image_size", "batch_size", "manifest_path", "provenance_path",
        "expected_source_manifest_sha256", "expected_portable_manifest_sha256",
        "expected_counts", "label_mapping", "excluded_original_classes",
    }, "data", drift)
    model = _expect_exact_keys(config.get("model"), {
        "constructor", "weights", "include_top", "include_preprocessing", "dropout",
        "output", "mixed_precision",
    }, "model", drift)
    augmentation = _expect_exact_keys(config.get("augmentation"), {
        "horizontal_flip", "rotation_factor", "zoom_factor", "translation_factor",
        "contrast_factor", "train_only",
    }, "augmentation", drift)
    training = _expect_exact_keys(config.get("training"), {
        "optimizer", "phase1", "phase2", "checkpoint_monitor", "checkpoint_mode",
        "restore_best_weights", "reduce_lr_on_plateau", "full_backbone_unfreeze_phase2",
        "freeze_batch_normalization_phase2",
    }, "training", drift)
    phase1 = _expect_exact_keys(training.get("phase1"), {
        "max_epochs", "learning_rate", "early_stopping_patience",
    }, "training.phase1", drift)
    phase2 = _expect_exact_keys(training.get("phase2"), {
        "max_epochs", "learning_rate", "early_stopping_patience",
    }, "training.phase2", drift)
    evaluation = _expect_exact_keys(config.get("evaluation"), {
        "primary_threshold", "threshold_min", "threshold_max", "threshold_step",
        "scientific_status", "holdout_access",
    }, "evaluation", drift)
    contract = _expect_exact_keys(config.get("contract"), {
        "reference_commit", "declared_frozen_protocol_sha256", "verification_status",
    }, "contract", drift)
    profile = config.get("profile")

    _expect(config.get("schema_version"), SCHEMA_VERSION, "schema_version", drift)
    if config.get("status") not in {"DRAFT_NOT_FROZEN", "DRAFT_NOT_APPROVED", "FROZEN"}:
        drift.append("status")
    _expect(config.get("approval_required"), True, "approval_required", drift)
    _expect(profile in {"baseline_v1", "pilot_v1"}, True, "profile", drift)
    _expect(project.get("seed"), 42, "seed", drift)
    _expect(project.get("output_root"), "outputs_v1", "output_root", drift)
    expected_prefix = (
        "mobilenetv3_small_baseline_v1" if profile == "baseline_v1"
        else "mobilenetv3_small_pilot_v1"
    )
    _expect(project.get("run_id_prefix"), expected_prefix, "run_id_prefix", drift)
    _expect(data.get("dataset_root_env"), "DATASET_ROOT", "dataset_root_env", drift)
    _expect(data.get("image_size"), 224, "image_size", drift)
    _expect(data.get("batch_size"), 16, "batch_size", drift)
    _expect(data.get("manifest_path"), "shared_data/manifest_binary_v1_portable.csv", "manifest_path", drift)
    _expect(data.get("provenance_path"), "shared_data/manifest_binary_v1_portable_provenance.json", "provenance_path", drift)
    _expect(data.get("expected_source_manifest_sha256"), SOURCE_MANIFEST_SHA256, "source_manifest_sha256", drift)
    _expect(data.get("label_mapping"), {"GOOD": 0, "BAD": 1}, "label_mapping", drift)
    _expect(data.get("excluded_original_classes"), ["green_orange"], "excluded_original_classes", drift)
    _expect(data.get("expected_counts"), EXPECTED_COUNTS, "expected_counts", drift)

    _expect(model.get("constructor"), "tf.keras.applications.MobileNetV3Small", "constructor", drift)
    _expect(model.get("weights"), "imagenet", "weights", drift)
    _expect(model.get("include_top"), False, "include_top", drift)
    _expect(model.get("include_preprocessing"), True, "include_preprocessing", drift)
    _expect(model.get("dropout"), 0.20, "dropout", drift)
    _expect(model.get("output"), "sigmoid_p_BAD", "model.output", drift)
    _expect(model.get("mixed_precision"), False, "mixed_precision", drift)

    expected_aug = {
        "horizontal_flip": True, "rotation_factor": 0.04, "zoom_factor": 0.10,
        "translation_factor": 0.05, "contrast_factor": 0.10, "train_only": True,
    }
    _expect(augmentation, expected_aug, "augmentation", drift)

    _expect(training.get("optimizer"), "Adam", "optimizer", drift)
    _expect(training.get("checkpoint_monitor"), "val_loss", "checkpoint_monitor", drift)
    _expect(training.get("checkpoint_mode"), "min", "checkpoint_mode", drift)
    _expect(training.get("restore_best_weights"), False, "restore_best_weights", drift)
    _expect(training.get("reduce_lr_on_plateau"), False, "reduce_lr_on_plateau", drift)
    _expect(training.get("full_backbone_unfreeze_phase2"), True, "full_backbone_unfreeze_phase2", drift)
    _expect(training.get("freeze_batch_normalization_phase2"), True, "freeze_batch_normalization_phase2", drift)
    expected_epochs = (10, 20) if profile == "baseline_v1" else (1, 1)
    _expect(phase1.get("max_epochs"), expected_epochs[0], "phase1.max_epochs", drift)
    _expect(phase2.get("max_epochs"), expected_epochs[1], "phase2.max_epochs", drift)
    _expect(phase1.get("learning_rate"), 0.001, "phase1.learning_rate", drift)
    _expect(phase2.get("learning_rate"), 0.0001, "phase2.learning_rate", drift)
    _expect(phase1.get("early_stopping_patience"), 3, "phase1.patience", drift)
    _expect(phase2.get("early_stopping_patience"), 5, "phase2.patience", drift)

    _expect(evaluation.get("primary_threshold"), 0.5, "primary_threshold", drift)
    _expect(evaluation.get("threshold_min"), 0.01, "threshold_min", drift)
    _expect(evaluation.get("threshold_max"), 0.99, "threshold_max", drift)
    _expect(evaluation.get("threshold_step"), 0.01, "threshold_step", drift)
    _expect(evaluation.get("scientific_status"), "LEGACY_DEV_INTERNAL_NOT_INDEPENDENT", "scientific_status", drift)
    _expect(evaluation.get("holdout_access"), "FORBIDDEN", "holdout_access", drift)
    _expect(contract.get("reference_commit"), REFERENCE_COMMIT, "reference_commit", drift)
    _expect(contract.get("declared_frozen_protocol_sha256"), FROZEN_PROTOCOL_SHA256, "protocol_sha256", drift)
    expected_verification = (
        "VERIFIED" if config.get("status") == "FROZEN"
        else "BLOCKED_MISSING_APPROVED_MANIFEST_AND_PROTOCOL_FILE"
    )
    _expect(contract.get("verification_status"), expected_verification, "verification_status", drift)

    portable_sha = data.get("expected_portable_manifest_sha256")
    if portable_sha is not None and not _HEX64.fullmatch(str(portable_sha).lower()):
        drift.append("expected_portable_manifest_sha256")
    if drift:
        raise ContractConfigError(f"V1 contract drift: {', '.join(drift)}")


def verify_frozen_config(config_path: str | Path, protocol_path: str | Path) -> VerifiedConfig:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise PreflightBlockedError(f"BLOCKED: config file missing: {path}")
    snapshot_bytes = path.read_bytes()
    try:
        config = yaml.safe_load(snapshot_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ContractConfigError("Benchmark config is not valid UTF-8 YAML.") from exc
    if not isinstance(config, dict):
        raise ContractConfigError("Benchmark config must be a YAML object.")
    validate_config(config)
    expected_name = "pilot_v1.yaml" if config["profile"] == "pilot_v1" else "baseline_v1.yaml"
    if path.parent.name != "configs" or path.name != expected_name:
        raise PreflightBlockedError("BLOCKED: config file identity/path is not canonical.")
    if config.get("status") != "FROZEN":
        raise PreflightBlockedError("BLOCKED: config status is not FROZEN.")
    if not config["data"].get("expected_portable_manifest_sha256"):
        raise PreflightBlockedError("BLOCKED: approved portable manifest SHA-256 is missing.")
    protocol = Path(protocol_path).resolve()
    if not protocol.is_file():
        raise PreflightBlockedError(f"BLOCKED: approved protocol file missing: {protocol}")
    actual_protocol_sha = sha256_bytes(protocol.read_bytes())
    if actual_protocol_sha != FROZEN_PROTOCOL_SHA256:
        raise PreflightBlockedError("BLOCKED: approved protocol SHA-256 mismatch.")
    return VerifiedConfig(
        path=path,
        repo_root=path.parent.parent,
        sha256=sha256_bytes(snapshot_bytes),
        protocol_path=protocol,
        protocol_sha256=actual_protocol_sha,
        profile=str(config["profile"]),
        values=_deep_freeze(config),
        snapshot_bytes=snapshot_bytes,
        _seal=_CONFIG_SEAL,
    )


def assert_ready_for_data_access(config: dict[str, Any]) -> None:
    """Compatibility preflight helper; it does not issue a VerifiedConfig."""
    validate_config(config)
    if config.get("status") != "FROZEN":
        raise PreflightBlockedError("BLOCKED: config is not FROZEN.")
    if not config["data"].get("expected_portable_manifest_sha256"):
        raise PreflightBlockedError("BLOCKED: approved portable manifest SHA-256 is missing.")
