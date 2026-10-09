# Binary V1 workflow threat model

Status: `PREFLIGHT_DESIGN_ONLY`

The supported execution path is
`orange.benchmark_v1.execute_approved_run(VerifiedRunContext)`. The context must
originate from frozen-config, manifest, and approval verification, and its files
and Git state are revalidated at execution boundaries.

The guards address accidental misuse inside the standard workflow: wrong stage,
stale hashes, dirty/wrong Git commit, arbitrary output roots, dataset/weight
binding drift, last-checkpoint evaluation, and artifact inconsistency.

They are not a security sandbox. A caller able to import private module names,
modify source, monkey-patch functions, inspect in-memory objects, or execute
arbitrary Python can bypass in-process conventions. `isinstance`, frozen
dataclasses, underscore-prefixed helpers, and module seals are not independent
approval evidence.

Approval authenticity requires an approval artifact plus its SHA-256 delivered
through an independent trusted channel. A hash calculated by the same untrusted
caller from the same artifact is not an approval. When that trust anchor is
missing, preflight remains `BLOCKED`.

`VERIFIED_RUN_CONTEXT` means only that config, protocol, manifest, approval, and
Git bindings were verified. It is not `PREFLIGHT_ACCEPTANCE` and does not mark
Gate A-D as collectively passed. Pilot and full execution remain separately
controlled by their stage-specific approval records.
