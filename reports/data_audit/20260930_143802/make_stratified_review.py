"""Create an additional class-stratified visual-review sample."""

from __future__ import annotations

import csv
import importlib.util
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit_data", ROOT / "audit_data.py")
audit_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(audit_data)


def main() -> None:
    with (ROOT / "near_duplicate_candidates.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    buckets = defaultdict(list)
    for row in rows:
        if row["review_status"] != "NEEDS_HUMAN_REVIEW":
            continue
        if row["original_class_a"] != row["original_class_b"]:
            continue
        buckets[(row["original_class_a"], int(row["phash_distance"]))].append(row)

    selected = []
    for class_name in sorted({key[0] for key in buckets}):
        for distance in (0, 2, 4):
            pool = sorted(buckets[(class_name, distance)], key=lambda row: row["pair_id"])
            if pool:
                # Pick early and middle candidates to avoid a single local sequence.
                indices = sorted({0, len(pool) // 2})
                for index in indices:
                    selected.append(pool[index])

    fields = list(selected[0].keys()) if selected else audit_data.NEAR_FIELDS
    with (ROOT / "stratified_visual_review_selection.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    sheet_dir = ROOT / "contact_sheets" / "stratified"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    created = audit_data.create_pair_sheets(selected, Path("E:/dataset"), sheet_dir)
    print(f"selected={len(selected)} sheets={len(created)}")
    print("pair_ids=" + ",".join(row["pair_id"] for row in selected))


if __name__ == "__main__":
    main()
