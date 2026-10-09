"""Synthetic/no-data tests for draft V1 configs."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from orange.benchmark_v1.config import (  # noqa: E402
    ContractConfigError,
    assert_ready_for_data_access,
    load_and_validate_config,
    validate_config,
    verify_frozen_config,
)


class BenchmarkConfigTests(unittest.TestCase):
    def test_baseline_and_pilot_configs_are_valid_drafts(self):
        baseline = load_and_validate_config(ROOT / "configs" / "baseline_v1.yaml")
        pilot = load_and_validate_config(ROOT / "configs" / "pilot_v1.yaml")
        self.assertEqual(baseline["training"]["phase1"]["max_epochs"], 10)
        self.assertEqual(baseline["training"]["phase2"]["max_epochs"], 20)
        self.assertEqual(pilot["training"]["phase1"]["max_epochs"], 1)
        self.assertEqual(pilot["training"]["phase2"]["max_epochs"], 1)

    def test_missing_approved_manifest_hash_blocks_data_access(self):
        config = load_and_validate_config(ROOT / "configs" / "baseline_v1.yaml")
        with self.assertRaisesRegex(ContractConfigError, "BLOCKED"):
            assert_ready_for_data_access(config)

    def test_contract_drift_is_rejected(self):
        config = load_and_validate_config(ROOT / "configs" / "baseline_v1.yaml")
        changed = copy.deepcopy(config)
        changed["training"]["phase2"]["learning_rate"] = 1e-5
        with self.assertRaises(ContractConfigError):
            validate_config(changed)

    def test_unreviewed_extra_config_field_is_rejected(self):
        config = load_and_validate_config(ROOT / "configs" / "baseline_v1.yaml")
        changed = copy.deepcopy(config)
        changed["training"]["caller_override"] = True
        with self.assertRaisesRegex(ContractConfigError, "schema"):
            validate_config(changed)

    def test_arbitrary_output_or_legacy_directory_is_rejected(self):
        config = load_and_validate_config(ROOT / "configs" / "baseline_v1.yaml")
        for unsafe in ("training/outputs", "../outputs_v1"):
            changed = copy.deepcopy(config)
            changed["project"]["output_root"] = unsafe
            with self.assertRaises(ContractConfigError):
                validate_config(changed)

    def test_draft_config_cannot_become_verified_config(self):
        with self.assertRaisesRegex(ContractConfigError, "BLOCKED"):
            verify_frozen_config(
                ROOT / "configs" / "baseline_v1.yaml",
                ROOT / "docs" / "PROTOCOL_EXCEPTION_REQUEST.md",
            )


if __name__ == "__main__":
    unittest.main()
