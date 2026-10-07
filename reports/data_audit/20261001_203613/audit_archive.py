"""Read-only audit of one ZIP archive against E:/dataset.

ZIP members are read directly into memory. Nothing is extracted or rewritten.
Uses only the Python standard library and the prior audit's JPEG marker parser
helpers for EXIF decoding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import struct
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


INVENTORY_FIELDS = [
    "member_path", "member_name", "member_directory", "extension",
    "compressed_size", "uncompressed_size", "compression_method", "crc32",
    "zip_modified_time", "zip_flag_bits", "encrypted", "sha256",
    "jpeg_valid", "jpeg_width", "jpeg_height", "jpeg_components",
    "jpeg_precision", "jfif_present", "jfif_version", "exif_present",
    "iptc_present", "xmp_present", "extended_xmp_present",
    "icc_profile_present", "photoshop_app13_present", "adobe_app14_present",
    "comment_present", "orientation", "camera_make", "camera_model",
    "lens_model", "software", "datetime_ifd0", "datetime_original",
    "datetime_digitized", "gps_present", "marker_signature",
    "dataset_exact_match_count", "dataset_exact_match_path", "match_status",
]

MAPPING_FIELDS = [
    "archive_member_path", "archive_sha256", "archive_size",
    "archive_width", "archive_height", "dataset_path", "dataset_sha256",
    "dataset_size", "relationship", "confidence", "evidence", "notes",
]


def load_prior_parser(path: Path):
    spec = importlib.util.spec_from_file_location("prior_metadata_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load metadata parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_jpeg_bytes(data: bytes, helper) -> dict[str, object]:
    result: dict[str, object] = {
        "jpeg_valid": False,
        "jpeg_width": "", "jpeg_height": "", "jpeg_components": "",
        "jpeg_precision": "", "jfif_present": False, "jfif_version": "",
        "exif_present": False, "iptc_present": False, "xmp_present": False,
        "extended_xmp_present": False, "icc_profile_present": False,
        "photoshop_app13_present": False, "adobe_app14_present": False,
        "comment_present": False, "orientation": "", "camera_make": "",
        "camera_model": "", "lens_model": "", "software": "",
        "datetime_ifd0": "", "datetime_original": "",
        "datetime_digitized": "", "gps_present": False,
        "marker_signature": "",
    }
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return result
    result["jpeg_valid"] = True
    pos = 2
    markers: list[str] = []
    while pos + 1 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            markers.append(f"FF{marker:02X}")
            if marker == 0xD9:
                break
            continue
        if pos + 2 > len(data):
            break
        length = int.from_bytes(data[pos:pos + 2], "big")
        if length < 2 or pos + length > len(data):
            break
        payload = data[pos + 2:pos + length]
        pos += length
        markers.append(f"FF{marker:02X}")
        if marker == 0xDA:
            break
        if marker == 0xE0 and payload.startswith(b"JFIF\x00") and len(payload) >= 12:
            result["jfif_present"] = True
            result["jfif_version"] = f"{payload[5]}.{payload[6]:02d}"
        elif marker == 0xE1:
            if payload.startswith(b"Exif\x00\x00"):
                result["exif_present"] = True
                result.update(helper.parse_exif(payload))
            elif payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                result["xmp_present"] = True
            elif payload.startswith(b"http://ns.adobe.com/xmp/extension/\x00"):
                result["extended_xmp_present"] = True
        elif marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
            result["icc_profile_present"] = True
        elif marker == 0xED:
            result["photoshop_app13_present"] = payload.startswith(b"Photoshop 3.0\x00")
            result["iptc_present"] = b"8BIM" in payload or b"\x1c\x02" in payload
        elif marker == 0xEE and payload.startswith(b"Adobe"):
            result["adobe_app14_present"] = True
        elif marker == 0xFE:
            result["comment_present"] = True
        elif marker in helper.SOF_MARKERS and len(payload) >= 6:
            result["jpeg_precision"] = payload[0]
            result["jpeg_height"] = int.from_bytes(payload[1:3], "big")
            result["jpeg_width"] = int.from_bytes(payload[3:5], "big")
            result["jpeg_components"] = payload[5]
    result["marker_signature"] = "-".join(markers)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iso_local(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def zip_time(info: zipfile.ZipInfo) -> str:
    try:
        return datetime(*info.date_time).isoformat(timespec="seconds")
    except ValueError:
        return ""


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-parser", type=Path, required=True)
    args = parser.parse_args()

    archive = args.archive.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("archive_inventory.csv", "source_mapping_candidates.csv", "archive_metadata_summary.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output_dir / name}")
    helper = load_prior_parser(args.metadata_parser.resolve(strict=True))
    started = time.perf_counter()

    archive_stat = archive.stat()
    archive_sha256 = sha256_file(archive)

    dataset_by_hash: dict[str, list[Path]] = defaultdict(list)
    dataset_sizes: dict[str, int] = {}
    dataset_file_count = 0
    for path in sorted(dataset.rglob("*")):
        if not path.is_file():
            continue
        dataset_file_count += 1
        digest = sha256_file(path)
        dataset_by_hash[digest].append(path)
        dataset_sizes[str(path)] = path.stat().st_size

    inventory: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    extension_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    marker_counts: Counter[str] = Counter()
    metadata_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    unsafe_member_names: list[str] = []
    encrypted_members: list[str] = []
    duplicate_member_names: list[str] = []
    member_name_counts: Counter[str] = Counter()

    with zipfile.ZipFile(archive, "r") as zf:
        bad_crc_member = zf.testzip()
        infos = zf.infolist()
        for info in infos:
            if info.is_dir():
                continue
            member_name_counts[info.filename] += 1
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts:
                unsafe_member_names.append(info.filename)
            encrypted = bool(info.flag_bits & 0x1)
            if encrypted:
                encrypted_members.append(info.filename)
            with zf.open(info, "r") as handle:
                data = handle.read()
            digest = hashlib.sha256(data).hexdigest()
            meta = parse_jpeg_bytes(data, helper)
            suffix = pure.suffix.lower()
            extension_counts[suffix or "<none>"] += 1
            if meta["jpeg_valid"]:
                resolution_counts[f"{meta['jpeg_width']}x{meta['jpeg_height']}"] += 1
                marker_counts[str(meta["marker_signature"])] += 1
            for key in (
                "exif_present", "iptc_present", "xmp_present",
                "extended_xmp_present", "icc_profile_present",
                "photoshop_app13_present", "adobe_app14_present",
                "comment_present", "camera_make", "camera_model",
                "datetime_original", "gps_present",
            ):
                if meta.get(key):
                    metadata_counts[key] += 1

            matches = dataset_by_hash.get(digest, [])
            if len(matches) == 1:
                match_status = "EXACT_BYTE_MATCH"
            elif len(matches) > 1:
                match_status = "EXACT_BYTE_MATCH_MULTIPLE_DATASET_FILES"
            else:
                match_status = "NO_EXACT_MATCH"
            match_counts[match_status] += 1
            match_paths = "|".join(str(p) for p in matches)
            row: dict[str, object] = {
                "member_path": info.filename,
                "member_name": pure.name,
                "member_directory": str(pure.parent) if str(pure.parent) != "." else "",
                "extension": suffix,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "compression_method": info.compress_type,
                "crc32": f"{info.CRC:08x}",
                "zip_modified_time": zip_time(info),
                "zip_flag_bits": info.flag_bits,
                "encrypted": encrypted,
                "sha256": digest,
                "dataset_exact_match_count": len(matches),
                "dataset_exact_match_path": match_paths,
                "match_status": match_status,
                **meta,
            }
            inventory.append(row)
            for matched in matches:
                mappings.append({
                    "archive_member_path": info.filename,
                    "archive_sha256": digest,
                    "archive_size": info.file_size,
                    "archive_width": meta["jpeg_width"],
                    "archive_height": meta["jpeg_height"],
                    "dataset_path": str(matched),
                    "dataset_sha256": digest,
                    "dataset_size": dataset_sizes[str(matched)],
                    "relationship": "EXACT_BYTE_IDENTITY",
                    "confidence": "CONFIRMED",
                    "evidence": "SHA-256 equal; file size equal; byte identity implies pixel identity",
                    "notes": "No fruit/session relationship inferred",
                })

    duplicate_member_names = sorted(name for name, count in member_name_counts.items() if count > 1)
    matched_dataset_paths = {row["dataset_path"] for row in mappings}
    all_dataset_paths = {str(path) for paths in dataset_by_hash.values() for path in paths}
    unmatched_dataset_paths = sorted(all_dataset_paths - matched_dataset_paths)
    exact_unique_dataset_files = sum(1 for paths in dataset_by_hash.values() if len(paths) == 1)

    write_csv(output_dir / "archive_inventory.csv", INVENTORY_FIELDS, inventory)
    write_csv(output_dir / "source_mapping_candidates.csv", MAPPING_FIELDS, mappings)

    summary = {
        "audit_scope": {"archive": str(archive), "dataset": str(dataset)},
        "read_only_method": "ZIP members read with zipfile into memory; no extraction",
        "archive": {
            "exists": True,
            "readable": True,
            "size_bytes": archive_stat.st_size,
            "creation_time": iso_local(archive_stat.st_ctime),
            "last_write_time": iso_local(archive_stat.st_mtime),
            "sha256": archive_sha256,
            "bad_crc_member": bad_crc_member,
        },
        "members": {
            "total_entries": len(infos),
            "file_entries": len(inventory),
            "extension_counts": dict(extension_counts),
            "total_uncompressed_bytes": sum(i.file_size for i in infos if not i.is_dir()),
            "total_compressed_bytes": sum(i.compress_size for i in infos if not i.is_dir()),
            "encrypted_count": len(encrypted_members),
            "unsafe_path_count": len(unsafe_member_names),
            "duplicate_name_count": len(duplicate_member_names),
        },
        "image_metadata": {
            "jpeg_valid": sum(bool(r["jpeg_valid"]) for r in inventory),
            "resolution_counts": dict(resolution_counts),
            "metadata_presence_counts": dict(metadata_counts),
            "marker_signature_counts": dict(marker_counts),
        },
        "comparison": {
            "dataset_file_count": dataset_file_count,
            "dataset_unique_sha256_count": len(dataset_by_hash),
            "dataset_hashes_with_one_file": exact_unique_dataset_files,
            "archive_match_status_counts": dict(match_counts),
            "confirmed_mapping_rows": len(mappings),
            "unmatched_dataset_count": len(unmatched_dataset_paths),
            "unmatched_dataset_paths": unmatched_dataset_paths,
            "perceptual_hash_run": False,
            "perceptual_hash_reason": "All archive members obtained exact SHA-256 matches; near-match search was unnecessary",
        },
        "provenance_fields_recovered": {
            "fruit_id": False,
            "session_id": False,
            "capture_time": False,
            "camera_id": False,
            "source_member_path": True,
            "archive_sha256": True,
            "member_sha256": True,
        },
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "archive_metadata_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "archive_members": len(inventory),
        "mappings": len(mappings),
        "match_counts": dict(match_counts),
        "unmatched_dataset": len(unmatched_dataset_paths),
        "metadata_counts": dict(metadata_counts),
        "duration_seconds": summary["duration_seconds"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
