"""Synthetic fixed-threshold and sweep tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))

from orange.benchmark_v1.metrics import calculate_metrics, threshold_sweep  # noqa: E402


class MetricsTests(unittest.TestCase):
    def test_primary_metrics_known_confusion(self):
        metrics = calculate_metrics([0, 0, 1, 1], [0.1, 0.8, 0.4, 0.9])
        self.assertEqual(metrics["threshold"], 0.5)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["fn_bad"], 1)
        self.assertEqual(metrics["fp_good"], 1)
        self.assertAlmostEqual(metrics["fpr_good"], 0.5)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)
        self.assertAlmostEqual(metrics["pr_auc"], 5 / 6)
        self.assertAlmostEqual(metrics["brier"], 0.255)

    def test_sweep_has_exactly_99_fixed_points(self):
        rows = threshold_sweep([0, 1], [0.2, 0.8])
        self.assertEqual(len(rows), 99)
        self.assertEqual(rows[0]["threshold"], 0.01)
        self.assertEqual(rows[49]["threshold"], 0.5)
        self.assertEqual(rows[-1]["threshold"], 0.99)
        self.assertEqual(len({row["threshold"] for row in rows}), 99)

    def test_label_drift_is_not_silently_coerced(self):
        for labels in ([0, 2], [False, 1], ["0", 1]):
            with self.assertRaises(ValueError):
                calculate_metrics(labels, [0.2, 0.8])

    def test_primary_threshold_drift_is_explicit(self):
        metrics = calculate_metrics([0, 1], [0.2, 0.8], threshold=0.4)
        self.assertEqual(metrics["threshold"], 0.4)


if __name__ == "__main__":
    unittest.main()
