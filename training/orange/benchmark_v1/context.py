"""Verifier-issued values for the standard Binary V1 workflow.

The seals catch accidental API misuse. They are deliberately not described as a
security sandbox: a caller that can import private names, alter source, or execute
arbitrary Python is outside the threat model. Approval authenticity still requires
an independently delivered trust anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

_CONFIG_SEAL = object()
_MANIFEST_SEAL = object()
_PARTITION_SEAL = object()
_APPROVAL_SEAL = object()
_RUN_SEAL = object()
_DATASET_SEAL = object()
_WEIGHT_SEAL = object()
_DIRECTORY_SEAL = object()
_CHECKPOINT_SEAL = object()


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class VerifiedConfig:
    path: Path
    repo_root: Path
    sha256: str
    protocol_path: Path
    protocol_sha256: str
    profile: str
    values: Mapping[str, Any]
    snapshot_bytes: bytes
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _CONFIG_SEAL:
            raise TypeError("VerifiedConfig can only be issued by verify_frozen_config().")


@dataclass(frozen=True)
class VerifiedRecord:
    relative_path: str
    original_class: str
    binary_label: str
    label_id: int
    split: str
    source_row_key: str


@dataclass(frozen=True)
class VerifiedTrainPartition:
    records: tuple[VerifiedRecord, ...]
    manifest_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _PARTITION_SEAL:
            raise TypeError("VerifiedTrainPartition is verifier-issued only.")
        if not self.records or any(row.split != "DEV_TRAIN" for row in self.records):
            raise ValueError("Verified train partition contains an invalid split.")


@dataclass(frozen=True)
class VerifiedValidationPartition:
    records: tuple[VerifiedRecord, ...]
    manifest_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _PARTITION_SEAL:
            raise TypeError("VerifiedValidationPartition is verifier-issued only.")
        if not self.records or any(row.split != "DEV_VALIDATION" for row in self.records):
            raise ValueError("Verified validation partition contains an invalid split.")


@dataclass(frozen=True)
class VerifiedManifest:
    path: Path
    provenance_path: Path
    sha256: str
    source_manifest_sha256: str
    provenance_sha256: str
    train: VerifiedTrainPartition
    validation: VerifiedValidationPartition
    counts: Mapping[str, Mapping[str, int]]
    holdout_metadata_count: int
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _MANIFEST_SEAL:
            raise TypeError("VerifiedManifest can only be issued by verify_manifest().")


@dataclass(frozen=True)
class VerifiedApproval:
    path: Path
    sha256: str
    approval_id: str
    approved_by: str
    approved_at_utc: str
    stage: str
    run_id: str
    source_commit: str
    config_sha256: str
    protocol_sha256: str
    manifest_sha256: str
    source_manifest_sha256: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _APPROVAL_SEAL:
            raise TypeError("VerifiedApproval can only be issued by verify_approval().")


@dataclass(frozen=True)
class VerifiedRunContext:
    config: VerifiedConfig
    manifest: VerifiedManifest
    approval: VerifiedApproval
    stage: str
    run_id: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _RUN_SEAL:
            raise TypeError("VerifiedRunContext can only be issued by build_run_context().")
        if self.stage != self.approval.stage or self.run_id != self.approval.run_id:
            raise ValueError("Run context and approval binding disagree.")


@dataclass(frozen=True)
class VerifiedTrainDataset:
    dataset: Any
    manifest_sha256: str
    sample_count: int
    batch_size: int
    sampling_strategy: str
    config_sha256: str
    image_size: int
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _DATASET_SEAL:
            raise TypeError("VerifiedTrainDataset is loader-issued only.")


@dataclass(frozen=True)
class VerifiedValidationDataset:
    dataset: Any
    manifest_sha256: str
    sample_count: int
    batch_size: int
    config_sha256: str
    image_size: int
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _DATASET_SEAL:
            raise TypeError("VerifiedValidationDataset is loader-issued only.")


@dataclass(frozen=True)
class VerifiedClassWeights:
    values: Mapping[int, float]
    manifest_sha256: str
    train_count: int
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _WEIGHT_SEAL:
            raise TypeError("VerifiedClassWeights is verifier-issued only.")
        if set(self.values) != {0, 1}:
            raise ValueError("Class weights must contain exactly GOOD=0 and BAD=1.")
        if self.train_count < 1 or any(float(value) <= 0 for value in self.values.values()):
            raise ValueError("Class weights and train_count must be positive.")
        object.__setattr__(self, "values", _deep_freeze(dict(self.values)))


@dataclass(frozen=True)
class IsolatedRunDirectory:
    path: Path
    run_id: str
    context_fingerprint: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _DIRECTORY_SEAL:
            raise TypeError("IsolatedRunDirectory is orchestrator-issued only.")


@dataclass(frozen=True)
class VerifiedCheckpoint:
    path: Path
    sha256: str
    kind: str
    run_id: str
    context_fingerprint: str
    _seal: object

    def __post_init__(self) -> None:
        if self._seal is not _CHECKPOINT_SEAL:
            raise TypeError("VerifiedCheckpoint is training verifier-issued only.")
        if self.kind not in {"GLOBAL_BEST", "LAST"}:
            raise ValueError("Checkpoint kind must be GLOBAL_BEST or LAST.")


def context_fingerprint(config_sha: str, protocol_sha: str, manifest_sha: str, approval_sha: str) -> str:
    import hashlib

    payload = "\n".join((config_sha, protocol_sha, manifest_sha, approval_sha)).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
