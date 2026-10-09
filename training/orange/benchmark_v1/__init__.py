"""Fail-closed Binary V1 workflow; legacy training remains untouched.

``execute_approved_run`` is the sole supported execution entrypoint. The package
guards against accidental workflow bypass, not arbitrary Python/source tampering.
"""

from .orchestrator import execute_approved_run, preflight

GOOD = 0
BAD = 1
PRIMARY_THRESHOLD = 0.5

__all__ = [
    "BAD", "GOOD", "PRIMARY_THRESHOLD", "execute_approved_run", "preflight",
]
