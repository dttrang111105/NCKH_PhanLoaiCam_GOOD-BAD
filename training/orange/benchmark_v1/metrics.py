"""Dependency-light Binary V1 metrics and fixed validation threshold sweep."""

from __future__ import annotations

from math import isfinite
from typing import Iterable

PRIMARY_THRESHOLD = 0.5


class MetricsContractError(ValueError):
    pass


def _validated(y_true: Iterable[int], p_bad: Iterable[float]) -> tuple[list[int], list[float]]:
    raw_labels = list(y_true)
    raw_probabilities = list(p_bad)
    if not raw_labels or len(raw_labels) != len(raw_probabilities):
        raise MetricsContractError("Labels/probabilities must have equal non-zero length.")
    if any(isinstance(value, bool) or value not in (0, 1) for value in raw_labels):
        raise MetricsContractError("Labels must be GOOD=0 or BAD=1.")
    labels = [int(value) for value in raw_labels]
    try:
        probabilities = [float(value) for value in raw_probabilities]
    except (TypeError, ValueError) as exc:
        raise MetricsContractError("p_BAD values must be numeric.") from exc
    if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
        raise MetricsContractError("p_BAD must contain finite probabilities in [0,1].")
    return labels, probabilities


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _roc_auc(labels: list[int], probabilities: list[float]) -> float:
    positives = [p for y, p in zip(labels, probabilities) if y == 1]
    negatives = [p for y, p in zip(labels, probabilities) if y == 0]
    if not positives or not negatives:
        raise MetricsContractError("ROC-AUC requires both classes.")
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _average_precision(labels: list[int], probabilities: list[float]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        raise MetricsContractError("PR-AUC requires at least one BAD sample.")
    groups: dict[float, list[int]] = {}
    for label, probability in zip(labels, probabilities):
        groups.setdefault(probability, []).append(label)
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    for probability in sorted(groups, reverse=True):
        group = groups[probability]
        tp += sum(group)
        fp += len(group) - sum(group)
        recall = tp / positive_count
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def calculate_metrics(
    y_true: Iterable[int], p_bad: Iterable[float], *, threshold: float = PRIMARY_THRESHOLD
) -> dict[str, object]:
    labels, probabilities = _validated(y_true, p_bad)
    if not 0.0 <= threshold <= 1.0:
        raise MetricsContractError("Threshold must be in [0,1].")
    predictions = [int(probability >= threshold) for probability in probabilities]
    tn = sum(y == 0 and pred == 0 for y, pred in zip(labels, predictions))
    fp = sum(y == 0 and pred == 1 for y, pred in zip(labels, predictions))
    fn = sum(y == 1 and pred == 0 for y, pred in zip(labels, predictions))
    tp = sum(y == 1 and pred == 1 for y, pred in zip(labels, predictions))
    precision_bad = tp / (tp + fp) if tp + fp else 0.0
    recall_bad = tp / (tp + fn) if tp + fn else 0.0
    precision_good = tn / (tn + fn) if tn + fn else 0.0
    recall_good = tn / (tn + fp) if tn + fp else 0.0
    f1_bad = _f1(precision_bad, recall_bad)
    f1_good = _f1(precision_good, recall_good)
    return {
        "threshold": float(threshold),
        "n": len(labels),
        "accuracy": (tn + tp) / len(labels),
        "precision_bad": precision_bad,
        "recall_bad": recall_bad,
        "f1_bad": f1_bad,
        "macro_f1": (f1_good + f1_bad) / 2.0,
        "roc_auc": _roc_auc(labels, probabilities),
        "pr_auc": _average_precision(labels, probabilities),
        "pr_auc_definition": "average_precision",
        "brier": sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels),
        "fn_bad": fn,
        "fp_good": fp,
        "fpr_good": fp / (fp + tn) if fp + tn else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def threshold_sweep(y_true: Iterable[int], p_bad: Iterable[float]) -> list[dict[str, object]]:
    labels, probabilities = _validated(y_true, p_bad)
    rows = []
    for index in range(1, 100):
        threshold = index / 100.0
        metrics = calculate_metrics(labels, probabilities, threshold=threshold)
        rows.append(
            {key: metrics[key] for key in (
                "threshold", "accuracy", "precision_bad", "recall_bad", "f1_bad",
                "macro_f1", "fn_bad", "fp_good", "fpr_good"
            )}
        )
    return rows
