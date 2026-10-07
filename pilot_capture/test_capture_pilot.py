"""Lightweight tests for pilot capture; no camera or GUI is opened."""

from __future__ import annotations

import csv
import inspect
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import capture_pilot as pilot


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame

    def read(self):
        return True, self.frame


class PilotCaptureTests(unittest.TestCase):
    def test_template_schema_is_current_and_empty(self):
        path = Path(__file__).with_name("manifest.csv")
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows, [pilot.FIELDS])

    def test_timestamp_is_bound_to_frame_before_prompt_logic(self):
        expected_frame = object()
        expected_time = datetime(2026, 10, 1, 12, 34, 56, 789000, tzinfo=timezone.utc)
        captured = pilot.receive_frame(
            FakeCamera(expected_frame), clock=lambda: expected_time
        )
        self.assertIs(captured.image, expected_frame)
        self.assertEqual(captured.host_received_at, expected_time)

        source = inspect.getsource(pilot.capture_fruit)
        self.assertLess(
            source.index("captured = receive_frame(camera)"),
            source.index("defect_visible = prompt_choice"),
        )
        self.assertIn("captured_at = captured.host_received_at", source)
        self.assertNotIn("datetime.now", source)

    def prompt_decision(self, answers: list[str]):
        with patch("builtins.input", side_effect=answers):
            return pilot.prompt_fruit({})

    def test_green_orange_allows_human_good_or_bad_decision(self):
        good = self.prompt_decision([
            "F_GREEN_GOOD", "APPROVED", "green_orange", "GOOD",
            "Quả thật đã chín và đạt chuẩn thương mại", "",
        ])
        bad = self.prompt_decision([
            "F_GREEN_BAD", "APPROVED", "green_orange", "BAD",
            "Quả thật chưa chín", "",
        ])
        self.assertEqual(good.original_class, "green_orange")
        self.assertEqual(bad.original_class, "green_orange")
        self.assertEqual(good.binary_label, "GOOD")
        self.assertEqual(bad.binary_label, "BAD")
        self.assertTrue(good.label_reason)
        self.assertTrue(bad.label_reason)

    def test_review_has_blank_binary_label(self):
        review = self.prompt_decision([
            "F_REVIEW", "REVIEW", "", "Chưa chắc độ chín", "",
        ])
        self.assertEqual(review.review_status, "REVIEW")
        self.assertEqual(review.binary_label, "")
        self.assertTrue(review.label_reason)

    def test_binary_label_storage_is_text(self):
        self.assertEqual(pilot.BINARY_LABELS, {"GOOD", "BAD"})
        self.assertIn("binary_label", pilot.FIELDS)
        self.assertNotIn("label", pilot.FIELDS)


if __name__ == "__main__":
    unittest.main()
