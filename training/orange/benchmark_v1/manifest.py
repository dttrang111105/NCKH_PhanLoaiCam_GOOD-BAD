"""Verified portable-manifest adapter with no split-generation fallback."""

from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .config import EXPECTED_COUNTS, SOURCE_MANIFEST_SHA256, sha256_bytes
from .context import (
    VerifiedConfig,
    VerifiedManifest,
    VerifiedRecord,
    VerifiedTrainPartition,
    VerifiedValidationPartition,
    _MANIFEST_SEAL,
    _PARTITION_SEAL,
)

GOOD_CLASS = "fresh_orange"
BAD_CLASSES = {"blackspotted_orange", "cankered_orange", "rotten_orange"}
ALLOWED_CLASSES = {GOOD_CLASS, *BAD_CLASSES}
REQUIRED_COLUMNS = {
    "relative_path", "original_class", "binary_label", "label_id", "split", "source_row_key",
}
REQUIRED_PROVENANCE = {
    "verification_status", "source_manifest_sha256", "portable_manifest_sha256",
    "transform_script_sha256", "row_counts", "label_policy_id", "created_at_utc",
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestContractError(ValueError):
    """Raised on any unverified provenance or manifest contract violation."""


def _assert_hex64(value: Any, name: str) -> str:
    text = str(value or "").lower()
    if not _HEX64.fullmatch(text):
        raise ManifestContractError(f"{name} must be 64 lowercase hex characters.")
    return text


def _validate_provenance(
    config: VerifiedConfig, provenance_path: Path, provenance_bytes: bytes, actual_sha: str
) -> str:
    if not provenance_path.is_file():
        raise ManifestContractError(f"BLOCKED: provenance file missing: {provenance_path}")
    try:
        data = json.loads(provenance_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestContractError("BLOCKED: provenance is unreadable or invalid JSON.") from exc
    missing = REQUIRED_PROVENANCE - set(data) if isinstance(data, dict) else REQUIRED_PROVENANCE
    if missing:
        raise ManifestContractError(f"Provenance fields missing: {sorted(missing)}")
    if set(data) != REQUIRED_PROVENANCE:
        raise ManifestContractError("Provenance contains unexpected fields.")
    if data["verification_status"] != "PASS":
        raise ManifestContractError("BLOCKED: provenance verification_status is not PASS.")
    source_sha = _assert_hex64(data["source_manifest_sha256"], "source_manifest_sha256")
    portable_sha = _assert_hex64(data["portable_manifest_sha256"], "portable_manifest_sha256")
    _assert_hex64(data["transform_script_sha256"], "transform_script_sha256")
    expected_portable = str(config.values["data"]["expected_portable_manifest_sha256"]).lower()
    if source_sha != SOURCE_MANIFEST_SHA256:
        raise ManifestContractError("Source manifest SHA-256 mismatch.")
    if portable_sha != expected_portable or actual_sha != expected_portable:
        raise ManifestContractError("Portable manifest SHA-256 mismatch.")
    if data["row_counts"] != EXPECTED_COUNTS:
        raise ManifestContractError("Provenance row_counts mismatch.")
    if data["label_policy_id"] != "NCKH_BACKBONE_BENCHMARK_V1":
        raise ManifestContractError("Unexpected label_policy_id.")
    created_at = str(data["created_at_utc"])
    if not created_at.endswith("Z"):
        raise ManifestContractError("created_at_utc must be explicit UTC ending in Z.")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestContractError("created_at_utc must be ISO-8601.") from exc
    if parsed_created_at.utcoffset() is None or parsed_created_at.utcoffset().total_seconds() != 0:
        raise ManifestContractError("created_at_utc must use UTC.")
    return source_sha


def _validate_relative_path(value: str, original_class: str) -> str:
    if not value or "\\" in value:
        raise ManifestContractError("relative_path must be non-empty POSIX form.")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] != original_class:
        raise ManifestContractError(f"Unsafe or class-mismatched relative_path: {value}")
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ManifestContractError(f"Unexpected image extension: {value}")
    return path.as_posix()


def _record_from_row(row: dict[str, str | None]) -> VerifiedRecord:
    try:
        original_class = str(row["original_class"] or "").strip()
        binary_label = str(row["binary_label"] or "").strip().upper()
        split = str(row["split"] or "").strip().upper()
        source_row_key = str(row["source_row_key"] or "").strip()
        relative_path = str(row["relative_path"] or "").strip()
        label_text = str(row["label_id"] or "").strip()
    except KeyError as exc:
        raise ManifestContractError(f"Manifest row missing field: {exc.args[0]}") from exc
    if original_class == "green_orange":
        raise ManifestContractError("green_orange is forbidden in Binary V1.")
    if original_class not in ALLOWED_CLASSES:
        raise ManifestContractError(f"Unknown original_class: {original_class}")
    expected_label = "GOOD" if original_class == GOOD_CLASS else "BAD"
    if binary_label != expected_label:
        raise ManifestContractError(f"Label drift for {original_class}.")
    if label_text not in {"0", "1"}:
        raise ManifestContractError("label_id must be exactly 0 or 1.")
    label_id = int(label_text)
    if label_id != (0 if binary_label == "GOOD" else 1):
        raise ManifestContractError("binary_label and label_id disagree.")
    if split not in EXPECTED_COUNTS:
        raise ManifestContractError(f"Unexpected split: {split}")
    if not source_row_key:
        raise ManifestContractError("source_row_key must be non-empty.")
    return VerifiedRecord(
        relative_path=_validate_relative_path(relative_path, original_class),
        original_class=original_class,
        binary_label=binary_label,
        label_id=label_id,
        split=split,
        source_row_key=source_row_key,
    )


def verify_manifest(config: VerifiedConfig) -> VerifiedManifest:
    """Issue verified train/validation partitions bound to a frozen config."""
    if not isinstance(config, VerifiedConfig):
        raise TypeError("verify_manifest requires VerifiedConfig.")
    data = config.values["data"]
    manifest_path = (config.repo_root / str(data["manifest_path"])).resolve()
    provenance_path = (config.repo_root / str(data["provenance_path"])).resolve()
    if config.repo_root not in manifest_path.parents or config.repo_root not in provenance_path.parents:
        raise ManifestContractError("Manifest/provenance path escapes repository root.")
    if not manifest_path.is_file():
        raise ManifestContractError(f"BLOCKED: portable manifest missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    provenance_bytes = provenance_path.read_bytes() if provenance_path.is_file() else b""
    actual_sha = sha256_bytes(manifest_bytes)
    source_sha = _validate_provenance(
        config, provenance_path, provenance_bytes, actual_sha
    )

    train: list[VerifiedRecord] = []
    validation: list[VerifiedRecord] = []
    counts = {split: Counter() for split in EXPECTED_COUNTS}
    seen_paths: set[str] = set()
    seen_keys: set[str] = set()
    try:
        manifest_text = manifest_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ManifestContractError("Portable manifest is not valid UTF-8 CSV.") from exc
    with io.StringIO(manifest_text, newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ManifestContractError(f"Manifest columns missing: {sorted(missing)}")
        if set(reader.fieldnames or []) != REQUIRED_COLUMNS:
            raise ManifestContractError("Manifest contains unexpected columns.")
        for raw_row in reader:
            record = _record_from_row(raw_row)
            if record.relative_path in seen_paths or record.source_row_key in seen_keys:
                raise ManifestContractError("Duplicate relative_path or source_row_key.")
            seen_paths.add(record.relative_path)
            seen_keys.add(record.source_row_key)
            counts[record.split][record.binary_label] += 1
            counts[record.split]["total"] += 1
            if record.split == "DEV_TRAIN":
                train.append(record)
            elif record.split == "DEV_VALIDATION":
                validation.append(record)
            # Holdout metadata is counted but its paths are not retained or resolved.

    normalized_counts = {split: dict(counter) for split, counter in counts.items()}
    if normalized_counts != EXPECTED_COUNTS:
        raise ManifestContractError(f"Frozen membership counts mismatch: {normalized_counts}")
    train_partition = VerifiedTrainPartition(tuple(train), actual_sha, _PARTITION_SEAL)
    validation_partition = VerifiedValidationPartition(tuple(validation), actual_sha, _PARTITION_SEAL)
    frozen_counts = MappingProxyType({k: MappingProxyType(dict(v)) for k, v in normalized_counts.items()})
    return VerifiedManifest(
        path=manifest_path,
        provenance_path=provenance_path,
        sha256=actual_sha,
        source_manifest_sha256=source_sha,
        provenance_sha256=sha256_bytes(provenance_bytes),
        train=train_partition,
        validation=validation_partition,
        counts=frozen_counts,
        holdout_metadata_count=normalized_counts["DEV_HOLDOUT"]["total"],
        _seal=_MANIFEST_SEAL,
    )


def resolve_verified_paths(
    partition: VerifiedTrainPartition | VerifiedValidationPartition,
    dataset_root: str | Path,
) -> tuple[Path, ...]:
    if not isinstance(partition, (VerifiedTrainPartition, VerifiedValidationPartition)):
        raise TypeError("Only verifier-issued train/validation partitions are accepted.")
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise ManifestContractError(f"DATASET_ROOT missing: {root}")
    resolved: list[Path] = []
    for row in partition.records:
        path = (root / PurePosixPath(row.relative_path)).resolve()
        if root not in path.parents:
            raise ManifestContractError(f"Path escapes DATASET_ROOT: {row.relative_path}")
        if not path.is_file():
            raise ManifestContractError(f"Dataset file missing: {row.relative_path}")
        resolved.append(path)
    return tuple(resolved)
