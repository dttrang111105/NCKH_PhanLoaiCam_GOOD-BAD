"""Read-only dataset audit for the orange MobileNetV3-Small project.

Creates inventory, exact-duplicate groups, cross-split pHash candidates and
contact sheets. It never modifies source images or the project manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
INVENTORY_FIELDS = [
    "relative_path", "manifest_path", "file_name", "original_class",
    "binary_label", "label", "split", "exists", "extension", "file_size",
    "file_mtime_utc", "has_jpeg_exif_segment", "decode_ok", "decode_error",
    "width", "height", "channels", "file_sha256", "pixel_sha256", "phash64",
]
EXACT_FIELDS = [
    "group_id", "match_basis", "hash", "member_count", "excess_images",
    "crosses_split", "split_set", "label_conflict", "member_index",
    "relative_path", "original_class", "binary_label", "split", "width", "height",
]
NEAR_FIELDS = [
    "pair_id", "relative_path_a", "original_class_a", "binary_label_a", "split_a",
    "relative_path_b", "original_class_b", "binary_label_b", "split_b",
    "phash_distance", "phash_similarity", "exact_relation", "review_status",
    "review_notes",
]


def write_csv(path: Path, fields: list[str], rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rel_to_dataset(path: Path, dataset_root: Path) -> str:
    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def phash64(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coeff = cv2.dct(np.float32(small))[:8, :8]
    flat = coeff.flatten()
    median = float(np.median(flat[1:]))
    bits = flat > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKNode:
    __slots__ = ("value", "indices", "children")

    def __init__(self, value: int, index: int):
        self.value = value
        self.indices = [index]
        self.children: dict[int, BKNode] = {}


class BKTree:
    def __init__(self):
        self.root: BKNode | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = BKNode(value, index)
            return
        node = self.root
        while True:
            distance = hamming(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = BKNode(value, index)
                return
            node = child

    def query(self, value: int, radius: int):
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming(value, node.value)
            if distance <= radius:
                for index in node.indices:
                    yield distance, index
            low, high = distance - radius, distance + radius
            for edge, child in node.children.items():
                if low <= edge <= high:
                    stack.append(child)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def scan_inventory(manifest_rows, dataset_root: Path):
    inventory = []
    total = len(manifest_rows)
    started = time.perf_counter()
    for position, source in enumerate(manifest_rows, 1):
        path = Path(source["image_path"])
        row = {
            "relative_path": rel_to_dataset(path, dataset_root),
            "manifest_path": str(path),
            "file_name": source.get("file_name", path.name),
            "original_class": source.get("original_class", ""),
            "binary_label": source.get("binary_label", ""),
            "label": source.get("label", ""),
            "split": source.get("split", ""),
            "exists": path.is_file(),
            "extension": path.suffix.lower(),
            "file_size": "",
            "file_mtime_utc": "",
            "has_jpeg_exif_segment": "",
            "decode_ok": False,
            "decode_error": "",
            "width": "",
            "height": "",
            "channels": "",
            "file_sha256": "",
            "pixel_sha256": "",
            "phash64": "",
        }
        if not path.is_file():
            row["decode_error"] = "FILE_MISSING"
            inventory.append(row)
            continue
        try:
            stat = path.stat()
            raw = path.read_bytes()
            row["file_size"] = stat.st_size
            row["file_mtime_utc"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)
            )
            row["file_sha256"] = hashlib.sha256(raw).hexdigest()
            row["has_jpeg_exif_segment"] = (
                path.suffix.lower() in {".jpg", ".jpeg"} and b"Exif\x00\x00" in raw[:65536]
            )
            # OpenCV IMREAD_COLOR applies JPEG EXIF orientation unless
            # IMREAD_IGNORE_ORIENTATION is explicitly supplied. Conversion to
            # RGB, shape and raw uint8 pixels define the decoded-pixel hash.
            bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode returned None")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            row.update(
                {
                    "decode_ok": True,
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "pixel_sha256": hashlib.sha256(
                        f"{height}x{width}x{channels}|uint8|RGB|".encode("ascii")
                        + np.ascontiguousarray(rgb).tobytes()
                    ).hexdigest(),
                    "phash64": f"{phash64(rgb):016x}",
                }
            )
        except Exception as exc:  # recorded rather than hiding partial coverage
            row["decode_error"] = f"{type(exc).__name__}: {exc}"
        inventory.append(row)
        if position % 500 == 0 or position == total:
            elapsed = time.perf_counter() - started
            print(f"[inventory] {position}/{total} ({elapsed:.1f}s)", flush=True)
    return inventory


def exact_groups(inventory):
    output = []
    summaries = {}
    group_number = 0
    for basis, field in (("FILE_BYTES", "file_sha256"), ("DECODED_RGB_PIXELS", "pixel_sha256")):
        buckets = defaultdict(list)
        for row in inventory:
            if row[field]:
                buckets[row[field]].append(row)
        groups = [members for members in buckets.values() if len(members) > 1]
        summaries[basis] = {
            "groups": len(groups),
            "images_in_groups": sum(len(group) for group in groups),
            "excess_images": sum(len(group) - 1 for group in groups),
            "cross_split_groups": sum(
                len({member["split"] for member in group}) > 1 for group in groups
            ),
            "cross_label_groups": sum(
                len({member["binary_label"] for member in group}) > 1 for group in groups
            ),
        }
        for members in groups:
            group_number += 1
            split_set = sorted({member["split"] for member in members})
            label_conflict = len({member["binary_label"] for member in members}) > 1
            for member_index, member in enumerate(members, 1):
                output.append(
                    {
                        "group_id": f"EXACT_{group_number:06d}",
                        "match_basis": basis,
                        "hash": member[field],
                        "member_count": len(members),
                        "excess_images": len(members) - 1,
                        "crosses_split": len(split_set) > 1,
                        "split_set": "|".join(split_set),
                        "label_conflict": label_conflict,
                        "member_index": member_index,
                        **member,
                    }
                )
    return output, summaries


def near_candidates(inventory, output_path: Path, radius: int, max_candidates: int):
    valid = [row for row in inventory if row["decode_ok"] and row["phash64"]]
    tree = BKTree()
    pair_count = 0
    distance_counts = Counter()
    split_pair_counts = Counter()
    label_conflicts = 0
    truncated = False
    selected: dict[int, dict] = {}
    closest: list[tuple[int, int, dict]] = []
    conflicts: list[tuple[int, int, dict]] = []
    boundary: list[tuple[int, int, dict]] = []
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEAR_FIELDS)
        writer.writeheader()
        for current_index, current in enumerate(valid):
            value = int(current["phash64"], 16)
            for distance, prior_index in tree.query(value, radius):
                prior = valid[prior_index]
                if prior["split"] == current["split"]:
                    continue
                if prior["pixel_sha256"] == current["pixel_sha256"]:
                    continue
                if pair_count >= max_candidates:
                    truncated = True
                    break
                pair_count += 1
                splits = tuple(sorted((prior["split"], current["split"])))
                conflict = prior["binary_label"] != current["binary_label"]
                row = {
                    "pair_id": f"NEAR_{pair_count:07d}",
                    "relative_path_a": prior["relative_path"],
                    "original_class_a": prior["original_class"],
                    "binary_label_a": prior["binary_label"],
                    "split_a": prior["split"],
                    "relative_path_b": current["relative_path"],
                    "original_class_b": current["original_class"],
                    "binary_label_b": current["binary_label"],
                    "split_b": current["split"],
                    "phash_distance": distance,
                    "phash_similarity": f"{1.0 - distance / 64.0:.6f}",
                    "exact_relation": "NONE",
                    "review_status": "NEEDS_HUMAN_REVIEW",
                    "review_notes": "pHash candidate only; not proof of same fruit or duplicate",
                }
                writer.writerow(row)
                distance_counts[distance] += 1
                split_pair_counts["|".join(splits)] += 1
                label_conflicts += int(conflict)
                closest.append((distance, pair_count, row))
                closest = sorted(closest, key=lambda item: (item[0], item[1]))[:12]
                if conflict:
                    conflicts.append((distance, pair_count, row))
                    conflicts = sorted(conflicts, key=lambda item: (item[0], item[1]))[:8]
                if distance >= max(0, radius - 1):
                    boundary.append((distance, pair_count, row))
                    boundary = sorted(boundary, key=lambda item: (-item[0], item[1]))[:10]
            tree.add(value, current_index)
            if truncated:
                break
            if (current_index + 1) % 1000 == 0:
                print(
                    f"[near] {current_index + 1}/{len(valid)}, candidates={pair_count}",
                    flush=True,
                )
    for _, _, row in closest + conflicts + boundary:
        selected[int(row["pair_id"].split("_")[1])] = row
    return {
        "candidate_pairs": pair_count,
        "radius": radius,
        "max_candidates": max_candidates,
        "truncated": truncated,
        "distance_counts": dict(sorted(distance_counts.items())),
        "split_pair_counts": dict(sorted(split_pair_counts.items())),
        "label_conflict_pairs": label_conflicts,
        "contact_sheet_pair_ids": [
            row["pair_id"] for _, row in sorted(selected.items())
        ],
    }, [row for _, row in sorted(selected.items())]


def load_rgb(dataset_root: Path, relative_path: str) -> np.ndarray | None:
    path = dataset_root / Path(relative_path)
    try:
        raw = path.read_bytes()
        bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def fit_thumb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, np.uint8)
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    resized = cv2.resize(
        rgb,
        (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def put_lines(canvas: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for line in lines:
        cv2.putText(
            canvas, line[:80], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (20, 20, 20), 1, cv2.LINE_AA,
        )
        y += 18


def create_pair_sheets(selected_rows, dataset_root: Path, sheet_dir: Path):
    created = []
    per_page = 4
    for page_start in range(0, len(selected_rows), per_page):
        page_rows = selected_rows[page_start:page_start + per_page]
        canvas = np.full((len(page_rows) * 310, 760, 3), 255, np.uint8)
        for row_index, row in enumerate(page_rows):
            y0 = row_index * 310
            for col, suffix in enumerate(("a", "b")):
                rgb = load_rgb(dataset_root, row[f"relative_path_{suffix}"])
                if rgb is not None:
                    canvas[y0 + 45:y0 + 285, col * 380:(col + 1) * 380] = fit_thumb(
                        rgb, 380, 240
                    )
                put_lines(
                    canvas,
                    [
                        row[f"relative_path_{suffix}"],
                        f"{row[f'original_class_{suffix}']} / {row[f'binary_label_{suffix}']} / {row[f'split_{suffix}']}",
                    ],
                    col * 380 + 5,
                    y0 + 17,
                )
            put_lines(
                canvas,
                [f"{row['pair_id']}  pHash distance={row['phash_distance']}"],
                5,
                y0 + 305,
            )
        output = sheet_dir / f"near_pairs_{page_start // per_page + 1:03d}.jpg"
        cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        created.append(output.name)
    return created


def create_class_sheets(inventory, dataset_root: Path, sheet_dir: Path, seed: int):
    rng = random.Random(seed)
    by_class_split = defaultdict(list)
    for row in inventory:
        if row["decode_ok"] and row["split"] in {"train", "validation"}:
            by_class_split[(row["original_class"], row["split"])].append(row)
    created = []
    selection_rows = []
    for class_name in sorted({key[0] for key in by_class_split}):
        chosen = []
        for split in ("train", "validation"):
            pool = sorted(by_class_split[(class_name, split)], key=lambda row: row["relative_path"])
            chosen.extend(rng.sample(pool, min(6, len(pool))))
        canvas = np.full((3 * 230, 4 * 260, 3), 255, np.uint8)
        for idx, row in enumerate(chosen[:12]):
            r, c = divmod(idx, 4)
            rgb = load_rgb(dataset_root, row["relative_path"])
            if rgb is not None:
                canvas[r * 230:r * 230 + 190, c * 260:(c + 1) * 260] = fit_thumb(rgb, 260, 190)
            put_lines(
                canvas,
                [Path(row["relative_path"]).name, f"{row['split']} / {row['binary_label']}"],
                c * 260 + 4,
                r * 230 + 205,
            )
            selection_rows.append(
                {
                    "original_class": class_name,
                    "relative_path": row["relative_path"],
                    "split": row["split"],
                    "binary_label": row["binary_label"],
                    "sampling_method": "seeded_random_6_train_6_validation_per_class",
                }
            )
        output = sheet_dir / f"class_sample_{class_name}.jpg"
        cv2.imwrite(str(output), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        created.append(output.name)
    return created, selection_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phash-radius", type=int, default=6)
    parser.add_argument("--max-near-candidates", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = args.output_dir / "contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_manifest(args.manifest)
    inventory = scan_inventory(manifest_rows, args.dataset_root)
    write_csv(args.output_dir / "dataset_inventory.csv", INVENTORY_FIELDS, inventory)

    manifest_paths = {str(Path(row["manifest_path"]).resolve()).lower() for row in inventory}
    actual_paths = {
        str(path.resolve()).lower()
        for path in args.dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    extra_paths = sorted(actual_paths - manifest_paths)
    missing_paths = [row["manifest_path"] for row in inventory if not row["exists"]]

    exact_rows, exact_summary = exact_groups(inventory)
    write_csv(args.output_dir / "exact_duplicate_groups.csv", EXACT_FIELDS, exact_rows)

    near_summary, selected_pairs = near_candidates(
        inventory,
        args.output_dir / "near_duplicate_candidates.csv",
        args.phash_radius,
        args.max_near_candidates,
    )
    pair_sheets = create_pair_sheets(selected_pairs, args.dataset_root, sheet_dir)
    class_sheets, class_selection = create_class_sheets(
        inventory, args.dataset_root, sheet_dir, args.seed
    )
    write_csv(
        args.output_dir / "class_sample_selection.csv",
        ["original_class", "relative_path", "split", "binary_label", "sampling_method"],
        class_selection,
    )

    resolution_counts = Counter(
        f"{row['width']}x{row['height']}"
        for row in inventory if row["decode_ok"]
    )
    extension_counts = Counter(row["extension"] for row in inventory)
    split_counts = Counter(row["split"] for row in inventory)
    class_counts = Counter(row["original_class"] for row in inventory)
    exif_counts = Counter(str(row["has_jpeg_exif_segment"]) for row in inventory)
    duration = time.perf_counter() - started
    summary = {
        "started_at": started_wall,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_seconds": duration,
        "manifest_rows": len(manifest_rows),
        "unique_manifest_paths": len(manifest_paths),
        "actual_image_files_under_dataset_root": len(actual_paths),
        "missing_manifest_files": len(missing_paths),
        "extra_image_files_not_in_manifest": len(extra_paths),
        "decode_failures": sum(not bool(row["decode_ok"]) for row in inventory),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "resolution_counts": dict(resolution_counts.most_common()),
        "jpeg_exif_segment_counts": dict(sorted(exif_counts.items())),
        "exact_duplicate_summary": exact_summary,
        "near_duplicate_summary": near_summary,
        "missing_paths": missing_paths,
        "extra_paths": extra_paths,
        "contact_sheets": pair_sheets + class_sheets,
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    methodology = {
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "file_exact": "SHA-256 of complete file bytes",
        "pixel_exact": (
            "SHA-256 of ASCII shape/dtype/colorspace prefix plus contiguous uint8 RGB pixels; "
            "decoded by cv2.imdecode(IMREAD_COLOR), which applies EXIF orientation by default"
        ),
        "near_duplicate": (
            "64-bit pHash: RGB->gray, INTER_AREA resize 32x32, OpenCV DCT, top-left 8x8, "
            "bits above median of 63 AC coefficients; BK-tree Hamming search; cross-split only; "
            "decoded-pixel exact matches excluded"
        ),
        "phash_radius": args.phash_radius,
        "phash_similarity_definition": "1 - HammingDistance/64",
        "near_candidate_limit": args.max_near_candidates,
        "class_sample": "Python random.Random(seed=42), 6 train + 6 validation per original class",
        "important_limit": (
            "pHash candidates are not proof of duplicate or fruit identity; pHash is not fully "
            "robust to arbitrary crop, rotation, occlusion or strong photometric transforms"
        ),
    }
    (args.output_dir / "audit_config.json").write_text(
        json.dumps(methodology, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
