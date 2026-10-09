"""Static backbone identity checks; no TensorFlow import or weights download."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _literal_keywords(call: ast.Call) -> dict[str, object]:
    values = {}
    for keyword in call.keywords:
        if keyword.arg is not None:
            try:
                values[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                pass
    return values


class ModelIdentityTests(unittest.TestCase):
    def test_legacy_builder_is_exact_mobilenetv3_small_contract(self):
        path = ROOT / "training" / "orange" / "model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        constructors = [
            call for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == "MobileNetV3Small"
        ]
        self.assertEqual(len(constructors), 1)
        keywords = _literal_keywords(constructors[0])
        self.assertIs(keywords["include_top"], False)
        self.assertEqual(keywords["weights"], "imagenet")
        self.assertIs(keywords["include_preprocessing"], True)

        dense_calls = [
            call for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == "Dense"
        ]
        self.assertTrue(any(
            call.args and ast.literal_eval(call.args[0]) == 1
            and _literal_keywords(call).get("activation") == "sigmoid"
            for call in dense_calls
        ))


if __name__ == "__main__":
    unittest.main()
