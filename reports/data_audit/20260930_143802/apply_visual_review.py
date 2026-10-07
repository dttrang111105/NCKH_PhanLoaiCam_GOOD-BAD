"""Apply the documented manual review of selected contact-sheet pairs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CANDIDATES = ROOT / "near_duplicate_candidates.csv"
GROUP_EVIDENCE = ROOT / "group_evidence.csv"

# Reviewed from contact_sheets/near_pairs_001.jpg ... near_pairs_008.jpg.
# CONFIRMED means same source image / same depicted fruit and framing with a
# small re-encode, noise, brightness or pixel-level variant. It is deliberately
# not called a recovered real-world fruit_id.
CONFIRMED = {
    "NEAR_0000002", "NEAR_0000003", "NEAR_0000004", "NEAR_0000005",
    "NEAR_0000007", "NEAR_0000008", "NEAR_0000012", "NEAR_0000014",
    "NEAR_0000015", "NEAR_0000143", "NEAR_0000236", "NEAR_0000557",
    "NEAR_0000681", "NEAR_0000709", "NEAR_0000733", "NEAR_0000821",
    "NEAR_0000848", "NEAR_0000888", "NEAR_0000924", "NEAR_0000937",
    "NEAR_0000949",
    # Additional class-stratified review in contact_sheets/stratified/.
    "NEAR_0008110", "NEAR_0008644", "NEAR_0007796", "NEAR_0008916",
    "NEAR_0008943", "NEAR_0010926", "NEAR_0014988", "NEAR_0010060",
    "NEAR_0015432", "NEAR_0010065", "NEAR_0001026", "NEAR_0004310",
    "NEAR_0000001", "NEAR_0000009", "NEAR_0024289", "NEAR_0027186",
    "NEAR_0024126", "NEAR_0024124", "NEAR_0029572", "NEAR_0021162",
    "NEAR_0022715", "NEAR_0021113", "NEAR_0022321", "NEAR_0021119",
    "NEAR_0022496",
}
REJECTED = {
    "NEAR_0007783", "NEAR_0010772", "NEAR_0010948", "NEAR_0010960",
    "NEAR_0011016", "NEAR_0011451", "NEAR_0011455", "NEAR_0011456",
    "NEAR_0007793", "NEAR_0015635", "NEAR_0029285",
}
UNCERTAIN = {"NEAR_0000006", "NEAR_0003035", "NEAR_0003598"}


def main() -> None:
    with CANDIDATES.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys()) if rows else []

    reviewed = []
    for row in rows:
        pair_id = row["pair_id"]
        if pair_id in CONFIRMED:
            row["review_status"] = "CONFIRMED_SAME_SOURCE_VARIANT"
            row["review_notes"] = (
                "Manual contact-sheet review: same depicted fruit/source framing; "
                "minor pixel/noise/brightness/re-encode variant. Not a fruit_id."
            )
            reviewed.append(row)
        elif pair_id in REJECTED:
            row["review_status"] = "REJECTED_VISUALLY_DIFFERENT"
            row["review_notes"] = (
                "Manual contact-sheet review: visually different fruit/image; pHash false positive "
                "driven by similar centered silhouette/background."
            )
            reviewed.append(row)
        elif pair_id in UNCERTAIN:
            row["review_status"] = "REVIEWED_UNCERTAIN"
            row["review_notes"] = (
                "Manual contact-sheet review inconclusive at sheet resolution; do not group."
            )
            reviewed.append(row)

    with CANDIDATES.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Build connected evidence groups only from confirmed reviewed pairs.
    graph = defaultdict(set)
    pair_rows = {}
    for row in reviewed:
        if row["review_status"] != "CONFIRMED_SAME_SOURCE_VARIANT":
            continue
        a, b = row["relative_path_a"], row["relative_path_b"]
        graph[a].add(b)
        graph[b].add(a)
        pair_rows[row["pair_id"]] = row

    component_for = {}
    component_number = 0
    for node in sorted(graph):
        if node in component_for:
            continue
        component_number += 1
        group_id = f"REVIEWED_DUPLICATE_GROUP_{component_number:03d}"
        queue = deque([node])
        component_for[node] = group_id
        while queue:
            current = queue.popleft()
            for neighbor in graph[current]:
                if neighbor not in component_for:
                    component_for[neighbor] = group_id
                    queue.append(neighbor)

    evidence_fields = [
        "evidence_group_id", "pair_id", "relative_path_a", "split_a",
        "original_class_a", "binary_label_a", "relative_path_b", "split_b",
        "original_class_b", "binary_label_b", "phash_distance", "status",
        "evidence_basis", "limitation",
    ]
    evidence = []
    for pair_id in sorted(pair_rows):
        row = pair_rows[pair_id]
        evidence.append(
            {
                "evidence_group_id": component_for[row["relative_path_a"]],
                "pair_id": pair_id,
                **row,
                "status": "CONFIRMED_SAME_SOURCE_VARIANT",
                "evidence_basis": "pHash candidate plus manual contact-sheet review",
                "limitation": (
                    "Pair/group is evidence of duplicate/source variant only; it is not a recovered "
                    "fruit_id, video_id or session_id, and the group is not exhaustive."
                ),
            }
        )
    with GROUP_EVIDENCE.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=evidence_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evidence)

    unique_images_by_split = defaultdict(set)
    for row in evidence:
        unique_images_by_split[row["split_a"]].add(row["relative_path_a"])
        unique_images_by_split[row["split_b"]].add(row["relative_path_b"])
    review_summary = {
        "contact_sheet_pairs_reviewed": len(reviewed),
        "confirmed_same_source_variant_pairs": len(CONFIRMED),
        "rejected_visually_different_pairs": len(REJECTED),
        "reviewed_uncertain_pairs": len(UNCERTAIN),
        "confirmed_evidence_groups": component_number,
        "unique_images_in_confirmed_groups": len(
            {path for paths in unique_images_by_split.values() for path in paths}
        ),
        "unique_confirmed_images_by_split": {
            split: len(paths) for split, paths in sorted(unique_images_by_split.items())
        },
        "confirmed_pairs_involving_test": sum(
            "test" in (row["split_a"], row["split_b"]) for row in evidence
        ),
        "review_scope": (
            "30 pairs selected as closest/conflicting/boundary candidates plus 30 additional "
            "same-class pairs stratified by five original classes and pHash distances 0/2/4"
        ),
        "interpretation_limit": (
            "Manual review confirms only same depicted source image/fruit framing. It does not "
            "recover real-world fruit_id, video_id or session_id, and groups are not exhaustive."
        ),
    }
    (ROOT / "visual_review_summary.json").write_text(
        json.dumps(review_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"reviewed={len(reviewed)} confirmed={len(CONFIRMED)} "
        f"rejected={len(REJECTED)} uncertain={len(UNCERTAIN)} "
        f"evidence_groups={component_number}"
    )


if __name__ == "__main__":
    main()
