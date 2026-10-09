"""Approval verification bound to frozen config/protocol/manifest hashes."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping
from pathlib import Path

from .config import sha256_bytes, sha256_file
from .context import (
    VerifiedApproval,
    VerifiedConfig,
    VerifiedManifest,
    VerifiedRunContext,
    _APPROVAL_SEAL,
    _RUN_SEAL,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
REQUIRED_FIELDS = {
    "schema_version", "status", "stage", "run_id", "approval_id", "approved_by",
    "approved_at_utc", "config_sha256", "protocol_sha256", "manifest_sha256",
    "source_manifest_sha256", "source_commit",
}


class ApprovalVerificationError(PermissionError):
    pass


@dataclass(frozen=True)
class ContextIntegrityEvidence:
    checked_at_utc: str
    source_commit: str
    dirty_tree: bool
    hashes: Mapping[str, str]


def _git_state(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain"], text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ApprovalVerificationError("Cannot verify repository Git state.") from exc
    return commit, dirty


def verify_approval(
    approval_path: str | Path,
    *,
    expected_approval_sha256: str,
    config: VerifiedConfig,
    manifest: VerifiedManifest,
) -> VerifiedApproval:
    """Verify an approval against an out-of-band trusted SHA-256 reference.

    The expected approval hash is the trust anchor. Supplying the hash from the
    untrusted approval file itself provides no authenticity and is forbidden by
    process policy, though Python alone cannot identify the human source of a value.
    """
    if not isinstance(config, VerifiedConfig) or not isinstance(manifest, VerifiedManifest):
        raise TypeError("Approval verification requires verified config and manifest.")
    expected_hash = str(expected_approval_sha256).lower()
    if not _HEX64.fullmatch(expected_hash):
        raise ApprovalVerificationError("Trusted approval SHA-256 must be 64 hex chars.")
    path = Path(approval_path).resolve()
    if not path.is_file():
        raise ApprovalVerificationError(f"Approval record missing: {path}")
    approval_bytes = path.read_bytes()
    actual_hash = sha256_bytes(approval_bytes)
    if actual_hash != expected_hash:
        raise ApprovalVerificationError("Approval record SHA-256 mismatch.")
    try:
        data = json.loads(approval_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApprovalVerificationError("Approval record is unreadable or invalid JSON.") from exc
    missing = REQUIRED_FIELDS - set(data) if isinstance(data, dict) else REQUIRED_FIELDS
    if missing:
        raise ApprovalVerificationError(f"Approval fields missing: {sorted(missing)}")
    if set(data) != REQUIRED_FIELDS:
        raise ApprovalVerificationError("Approval record contains unexpected fields.")
    if data["schema_version"] != "NCKH_BASELINE_V1_APPROVAL" or data["status"] != "APPROVED":
        raise ApprovalVerificationError("Approval schema/status is invalid.")
    stage = str(data["stage"])
    required_stage = "PILOT" if config.profile == "pilot_v1" else "FULL"
    if stage != required_stage:
        raise ApprovalVerificationError("Pilot/full config and approval stage mismatch.")
    run_id = str(data["run_id"])
    prefix = str(config.values["project"]["run_id_prefix"])
    if not _RUN_ID.fullmatch(run_id) or not run_id.startswith(prefix + "_"):
        raise ApprovalVerificationError("Approval run_id is invalid for this config.")
    bindings = {
        "config_sha256": config.sha256,
        "protocol_sha256": config.protocol_sha256,
        "manifest_sha256": manifest.sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
    }
    drift = [key for key, expected in bindings.items() if str(data[key]).lower() != expected]
    if drift:
        raise ApprovalVerificationError(f"Approval hash binding mismatch: {', '.join(drift)}")
    if not str(data["approval_id"]).strip() or not str(data["approved_by"]).strip():
        raise ApprovalVerificationError("Approval identity fields must be non-empty.")
    approved_at = str(data["approved_at_utc"])
    if not approved_at.endswith("Z"):
        raise ApprovalVerificationError("approved_at_utc must be explicit UTC ending in Z.")
    try:
        datetime.fromisoformat(approved_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ApprovalVerificationError("approved_at_utc must be valid ISO-8601 UTC.") from exc
    actual_commit, dirty = _git_state(config.repo_root)
    if str(data["source_commit"]) != actual_commit or dirty:
        raise ApprovalVerificationError("Approval source commit mismatch or worktree is dirty.")
    return VerifiedApproval(
        path=path,
        sha256=actual_hash,
        approval_id=str(data["approval_id"]),
        approved_by=str(data["approved_by"]),
        approved_at_utc=str(data["approved_at_utc"]),
        stage=stage,
        run_id=run_id,
        source_commit=actual_commit,
        config_sha256=config.sha256,
        protocol_sha256=config.protocol_sha256,
        manifest_sha256=manifest.sha256,
        source_manifest_sha256=manifest.source_manifest_sha256,
        _seal=_APPROVAL_SEAL,
    )


def build_run_context(
    *, config: VerifiedConfig, manifest: VerifiedManifest, approval: VerifiedApproval
) -> VerifiedRunContext:
    if not isinstance(config, VerifiedConfig) or not isinstance(manifest, VerifiedManifest):
        raise TypeError("Run context requires verifier-issued config and manifest.")
    if not isinstance(approval, VerifiedApproval):
        raise TypeError("Run context requires verifier-issued approval.")
    if (
        approval.config_sha256 != config.sha256
        or approval.protocol_sha256 != config.protocol_sha256
        or approval.manifest_sha256 != manifest.sha256
        or approval.source_manifest_sha256 != manifest.source_manifest_sha256
    ):
        raise ApprovalVerificationError("Verified approval is not bound to this config/manifest.")
    return VerifiedRunContext(
        config=config,
        manifest=manifest,
        approval=approval,
        stage=approval.stage,
        run_id=approval.run_id,
        _seal=_RUN_SEAL,
    )


def revalidate_run_context(context: VerifiedRunContext) -> ContextIntegrityEvidence:
    """Recheck immutable inputs and Git state at an execution boundary.

    This protects against stale files in the standard workflow. It does not make
    arbitrary in-process Python execution trustworthy.
    """
    if not isinstance(context, VerifiedRunContext):
        raise TypeError("Context revalidation requires VerifiedRunContext.")
    config, manifest, approval = context.config, context.manifest, context.approval
    checks = {
        "config_sha256": (config.path, config.sha256),
        "protocol_sha256": (config.protocol_path, config.protocol_sha256),
        "manifest_sha256": (manifest.path, manifest.sha256),
        "provenance_sha256": (manifest.provenance_path, manifest.provenance_sha256),
        "approval_sha256": (approval.path, approval.sha256),
    }
    actual: dict[str, str] = {}
    for name, (path, expected) in checks.items():
        if not path.is_file():
            raise ApprovalVerificationError(f"Stale run context: missing {name} file.")
        value = sha256_file(path)
        if value != expected:
            raise ApprovalVerificationError(f"Stale run context: {name} mismatch.")
        actual[name] = value
    if sha256_bytes(config.snapshot_bytes) != config.sha256:
        raise ApprovalVerificationError("Verified config snapshot/hash mismatch.")
    commit, dirty = _git_state(config.repo_root)
    if commit != approval.source_commit or dirty:
        raise ApprovalVerificationError("Stale run context: Git HEAD changed or worktree is dirty.")
    build_run_context(config=config, manifest=manifest, approval=approval)
    return ContextIntegrityEvidence(
        checked_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        source_commit=commit,
        dirty_tree=dirty,
        hashes=MappingProxyType(actual),
    )
