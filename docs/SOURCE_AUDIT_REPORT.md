# MobileNetV3-Small Binary V1 — source audit and implementation status

Date: 2026-10-09 (Asia/Bangkok)

Personal repository audit commit: `79ba6ad59ff281cc9dce6b6a4d36406d78bdce35`

Group contract reference commit: `addaee99f8ba410ac7efd1e6abce34c042561c75`

## Status

`INCOMPLETE / BLOCKED_MISSING_APPROVED_MANIFEST_AND_PROTOCOL_FILE`

No training, pilot, real-image loader, prediction, or holdout evaluation was run while preparing this implementation.

## Kept unchanged

- Legacy commercial policy and `training/config.yaml`.
- Legacy `training/main.py`, `training/orange/data.py`, `training/orange/model.py`, and `training/orange/evaluation.py`.
- All historical checkpoints, split CSVs, reports, plots, and TFLite artifacts.
- The existing MobileNetV3-Small constructor: ImageNet, `include_top=False`, embedded preprocessing, GAP, Dropout, sigmoid `p_BAD`.

Legacy outputs remain `LEGACY_NON_COMPARABLE` for the shared Binary V1 benchmark.

## Added benchmark-only surface

- Draft configs under `configs/`; neither is frozen or approved.
- Sealed `VerifiedConfig`, manifest/partition, approval, run-context, dataset,
  class-weight, run-directory, and checkpoint values. These prevent accidental API
  bypass; they are explicitly not a security boundary against arbitrary Python code.
- Fail-closed portable-manifest adapter with exact labels/classes/splits/hash/provenance
  checks and no holdout partition exposed to the loader.
- Stage-specific approval verification using an out-of-band approval SHA-256 trust
  anchor, plus config/protocol/manifest/source-manifest/Git-commit bindings.
- Unified preflight/orchestrator flow. Preflight cannot construct a model or invoke
  training, and execution cannot accept caller-provided datasets, weights, output
  paths, or checkpoint paths.
- Two-phase training contract with train-only class weights, P1 backbone freeze, P2
  full unfreeze with BatchNorm frozen, a new optimizer, global `val_loss` best
  tracking, distinct best/last checkpoints, and SHA-verified global-best reload for
  validation evaluation.
- Prediction/metrics/threshold/artifact validators with cross-artifact run ID,
  manifest membership, label, checkpoint SHA, and count consistency checks.
- Stale-context revalidation checks Git HEAD/dirty state and config, protocol,
  manifest, provenance, and approval hashes before directory creation, immediately
  before training, and before artifact finalization.
- The supported execution API is `execute_approved_run`; internal Python seals and
  `isinstance` checks are workflow guards, not independent approval evidence or a
  sandbox. See `BINARY_V1_THREAT_MODEL.md`.
- Synthetic negative tests for forged approval, arbitrary partitions/datasets,
  manifest/provenance drift, wrong weights, stage drift, unsafe output roots,
  last-as-best evaluation, forged output bindings, threshold/label drift, missing
  fields, and cross-artifact mismatch.

## Hardening verification

`python -m unittest discover -s tests -p "test_benchmark_v1_*.py" -v`

Result on 2026-10-09: `PASS` (50 synthetic/static tests). This is not an
integration, real-data, model-construction, pilot, or training PASS.

The contract-level executed-run status remains `INCOMPLETE` when independent
holdout-access instrumentation is absent, with `UNVERIFIED` recorded in gate
details. It does not manufacture zero-access counters or convert training
completion into a hard-coded PASS.

## Unresolved blockers

1. Approved portable manifest CSV is not available.
2. Approved portable manifest SHA-256 is not available; draft configs intentionally store `null`.
3. The group documentation declares frozen protocol SHA-256 `6cedb8cf...`, while the checked-out repository copy has different file bytes. The exact approved frozen file must be supplied or reconciled.
4. Model construction/forward smoke remains `NOT_RUN`; no pretrained weights were downloaded.
5. Pilot and full training remain forbidden until the relevant gates and approvals are recorded.

## Protocol exception status

No exception was silently adopted. The protocol-file hash discrepancy requires clarification; if the checked-out protocol copy is intended to replace the declared frozen artifact, record a `PROTOCOL_EXCEPTION_REQUEST` or publish an approved correction before pilot.
