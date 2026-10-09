"""Synthetic manifest/context negative tests; no project dataset is used."""

from __future__ import annotations

import sys
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "tests"))

from benchmark_v1_fixtures import verified_fixture  # noqa: E402
from orange.benchmark_v1.context import VerifiedRecord, VerifiedTrainPartition  # noqa: E402
from orange.benchmark_v1.manifest import ManifestContractError, _record_from_row, verify_manifest  # noqa: E402


class ManifestTests(unittest.TestCase):
    def test_verified_fixture_issues_only_train_validation_partitions(self):
        with verified_fixture() as fixture:
            manifest = fixture.context.manifest
            self.assertEqual({row.split for row in manifest.train.records}, {"DEV_TRAIN"})
            self.assertEqual({row.split for row in manifest.validation.records}, {"DEV_VALIDATION"})
            self.assertFalse(hasattr(manifest, "holdout"))
            self.assertEqual(manifest.holdout_metadata_count, 3)

    def test_arbitrary_record_cannot_construct_verified_partition(self):
        row = VerifiedRecord("fresh_orange/x.jpg", "fresh_orange", "GOOD", 0, "DEV_TRAIN", "x")
        with self.assertRaises(TypeError):
            VerifiedTrainPartition((row,), "a" * 64, object())

    def test_changed_manifest_fails_hash_binding(self):
        with verified_fixture() as fixture:
            fixture.manifest_path.write_text("forged", encoding="utf-8")
            with self.assertRaisesRegex(ManifestContractError, "mismatch"):
                verify_manifest(fixture.context.config)

    def test_wrong_provenance_is_rejected(self):
        with verified_fixture() as fixture:
            data = json.loads(fixture.provenance_path.read_text(encoding="utf-8"))
            data["verification_status"] = "UNVERIFIED"
            fixture.provenance_path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ManifestContractError, "provenance"):
                verify_manifest(fixture.context.config)

    def test_label_drift_is_rejected(self):
        bad = {
            "relative_path": "fresh_orange/a.jpg", "original_class": "fresh_orange",
            "binary_label": "BAD", "label_id": "1", "split": "DEV_TRAIN",
            "source_row_key": "x",
        }
        with self.assertRaisesRegex(ManifestContractError, "Label drift"):
            _record_from_row(bad)

    def test_verified_config_is_deeply_immutable(self):
        with verified_fixture() as fixture:
            with self.assertRaises(TypeError):
                fixture.context.config.values["training"]["phase1"]["max_epochs"] = 1


if __name__ == "__main__":
    unittest.main()
