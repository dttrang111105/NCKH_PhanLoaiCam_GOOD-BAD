"""Read-only JPEG metadata audit for E:/dataset.

Uses only the Python standard library. It parses JPEG markers and common EXIF
fields without decoding or rewriting image pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import struct
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


FIELDS = [
    "relative_path", "file_name", "original_class", "binary_label", "split",
    "file_size", "windows_creation_time", "last_write_time", "extension",
    "jpeg_valid", "jpeg_width", "jpeg_height", "jpeg_components", "jpeg_precision",
    "jfif_present", "jfif_version", "jfif_density_units", "jfif_x_density",
    "jfif_y_density", "exif_present", "iptc_present", "xmp_present",
    "extended_xmp_present", "icc_profile_present", "icc_profile_chunks",
    "photoshop_app13_present", "adobe_app14_present", "comment_present",
    "comment_text", "orientation", "camera_make", "camera_model", "lens_model",
    "software", "datetime_ifd0", "datetime_original", "datetime_digitized",
    "offset_time", "offset_time_original", "offset_time_digitized", "gps_present",
    "xmp_create_date", "xmp_modify_date", "xmp_metadata_date", "xmp_creator_tool",
    "xmp_camera_make", "xmp_camera_model", "metadata_fields_available",
    "zone_identifier_present", "zone_id", "zone_referrer_url", "zone_host_url",
    "capture_identity_confidence", "notes",
]


TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 9: 4, 10: 8}
SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
}


def local_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def tiff_value(tiff: bytes, endian: str, field_type: int, count: int, raw4: bytes):
    size = TYPE_SIZES.get(field_type)
    if size is None or count < 0 or size * count > 16_000_000:
        return None
    total = size * count
    if total <= 4:
        data = raw4[:total]
    else:
        offset = struct.unpack(endian + "I", raw4)[0]
        if offset < 0 or offset + total > len(tiff):
            return None
        data = tiff[offset:offset + total]
    try:
        if field_type == 2:
            return data.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()
        if field_type == 3:
            values = struct.unpack(endian + f"{count}H", data)
            return values[0] if count == 1 else list(values)
        if field_type == 4:
            values = struct.unpack(endian + f"{count}I", data)
            return values[0] if count == 1 else list(values)
        if field_type == 9:
            values = struct.unpack(endian + f"{count}i", data)
            return values[0] if count == 1 else list(values)
        if field_type in {1, 7}:
            return data[0] if count == 1 else data.hex()
    except (struct.error, ValueError):
        return None
    return None


def parse_ifd(tiff: bytes, endian: str, offset: int) -> dict[int, object]:
    result: dict[int, object] = {}
    if offset < 0 or offset + 2 > len(tiff):
        return result
    try:
        count = struct.unpack_from(endian + "H", tiff, offset)[0]
    except struct.error:
        return result
    if count > 4096 or offset + 2 + count * 12 > len(tiff):
        return result
    for index in range(count):
        pos = offset + 2 + index * 12
        try:
            tag, field_type, item_count = struct.unpack_from(endian + "HHI", tiff, pos)
        except struct.error:
            continue
        value = tiff_value(tiff, endian, field_type, item_count, tiff[pos + 8:pos + 12])
        if value is not None:
            result[tag] = value
    return result


def parse_exif(payload: bytes) -> dict[str, object]:
    result: dict[str, object] = {"exif_parse_ok": False}
    if not payload.startswith(b"Exif\x00\x00"):
        return result
    tiff = payload[6:]
    if len(tiff) < 8 or tiff[:2] not in {b"II", b"MM"}:
        return result
    endian = "<" if tiff[:2] == b"II" else ">"
    try:
        if struct.unpack_from(endian + "H", tiff, 2)[0] != 42:
            return result
        ifd0_offset = struct.unpack_from(endian + "I", tiff, 4)[0]
    except struct.error:
        return result
    ifd0 = parse_ifd(tiff, endian, ifd0_offset)
    exif_ifd = parse_ifd(tiff, endian, int(ifd0.get(0x8769, -1)))
    gps_ifd = parse_ifd(tiff, endian, int(ifd0.get(0x8825, -1)))
    result.update(
        {
            "exif_parse_ok": True,
            "orientation": ifd0.get(0x0112, ""),
            "camera_make": ifd0.get(0x010F, ""),
            "camera_model": ifd0.get(0x0110, ""),
            "software": ifd0.get(0x0131, ""),
            "datetime_ifd0": ifd0.get(0x0132, ""),
            "datetime_original": exif_ifd.get(0x9003, ""),
            "datetime_digitized": exif_ifd.get(0x9004, ""),
            "offset_time": exif_ifd.get(0x9010, ""),
            "offset_time_original": exif_ifd.get(0x9011, ""),
            "offset_time_digitized": exif_ifd.get(0x9012, ""),
            "lens_model": exif_ifd.get(0xA434, ""),
            "gps_present": bool(gps_ifd),
        }
    )
    return result


def xmp_field(text: str, names: list[str]) -> str:
    for name in names:
        escaped = re.escape(name)
        patterns = [
            rf'{escaped}\s*=\s*["\']([^"\']+)["\']',
            rf'<{escaped}[^>]*>([^<]+)</{escaped}>',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
    return ""


def parse_jpeg(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "jpeg_valid": False,
        "jpeg_width": "", "jpeg_height": "", "jpeg_components": "",
        "jpeg_precision": "", "jfif_present": False, "jfif_version": "",
        "jfif_density_units": "", "jfif_x_density": "", "jfif_y_density": "",
        "exif_present": False, "iptc_present": False, "xmp_present": False,
        "extended_xmp_present": False, "icc_profile_present": False,
        "icc_profile_chunks": 0, "photoshop_app13_present": False,
        "adobe_app14_present": False, "comment_present": False, "comment_text": "",
        "orientation": "", "camera_make": "", "camera_model": "", "lens_model": "",
        "software": "", "datetime_ifd0": "", "datetime_original": "",
        "datetime_digitized": "", "offset_time": "", "offset_time_original": "",
        "offset_time_digitized": "", "gps_present": False, "xmp_create_date": "",
        "xmp_modify_date": "", "xmp_metadata_date": "", "xmp_creator_tool": "",
        "xmp_camera_make": "", "xmp_camera_model": "", "marker_signature": "",
    }
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return result
    result["jpeg_valid"] = True
    pos = 2
    markers = []
    comments = []
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
        if marker == 0xDA:  # Start of Scan; metadata markers precede entropy data.
            break
        if marker == 0xE0 and payload.startswith(b"JFIF\x00") and len(payload) >= 12:
            result["jfif_present"] = True
            result["jfif_version"] = f"{payload[5]}.{payload[6]:02d}"
            result["jfif_density_units"] = payload[7]
            result["jfif_x_density"] = int.from_bytes(payload[8:10], "big")
            result["jfif_y_density"] = int.from_bytes(payload[10:12], "big")
        elif marker == 0xE1:
            if payload.startswith(b"Exif\x00\x00"):
                result["exif_present"] = True
                result.update(parse_exif(payload))
            elif payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00"):
                result["xmp_present"] = True
                text = payload.split(b"\x00", 1)[-1].decode("utf-8", errors="replace")
                result["xmp_create_date"] = xmp_field(text, ["xmp:CreateDate", "photoshop:DateCreated", "exif:DateTimeOriginal"])
                result["xmp_modify_date"] = xmp_field(text, ["xmp:ModifyDate"])
                result["xmp_metadata_date"] = xmp_field(text, ["xmp:MetadataDate"])
                result["xmp_creator_tool"] = xmp_field(text, ["xmp:CreatorTool"])
                result["xmp_camera_make"] = xmp_field(text, ["tiff:Make"])
                result["xmp_camera_model"] = xmp_field(text, ["tiff:Model"])
            elif payload.startswith(b"http://ns.adobe.com/xmp/extension/\x00"):
                result["extended_xmp_present"] = True
        elif marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"):
            result["icc_profile_present"] = True
            result["icc_profile_chunks"] = int(result["icc_profile_chunks"]) + 1
        elif marker == 0xED:
            result["photoshop_app13_present"] = payload.startswith(b"Photoshop 3.0\x00")
            result["iptc_present"] = b"8BIM" in payload or b"\x1c\x02" in payload
        elif marker == 0xEE and payload.startswith(b"Adobe"):
            result["adobe_app14_present"] = True
        elif marker == 0xFE:
            result["comment_present"] = True
            comments.append(payload.decode("utf-8", errors="replace").strip())
        elif marker in SOF_MARKERS and len(payload) >= 6:
            result["jpeg_precision"] = payload[0]
            result["jpeg_height"] = int.from_bytes(payload[1:3], "big")
            result["jpeg_width"] = int.from_bytes(payload[3:5], "big")
            result["jpeg_components"] = payload[5]
    result["comment_text"] = " | ".join(comments)[:500]
    result["marker_signature"] = "-".join(markers)
    return result


def parse_zone_identifier(path: Path) -> dict[str, object]:
    """Read the NTFS Zone.Identifier alternate stream when present."""
    result: dict[str, object] = {
        "zone_identifier_present": False,
        "zone_id": "",
        "zone_referrer_url": "",
        "zone_host_url": "",
    }
    try:
        text = Path(str(path) + ":Zone.Identifier").read_text(
            encoding="utf-8", errors="replace"
        ).replace("\x00", "")
    except (FileNotFoundError, OSError):
        return result
    result["zone_identifier_present"] = True
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        normalized = key.strip().lower()
        if normalized == "zoneid":
            result["zone_id"] = value.strip()
        elif normalized == "referrerurl":
            result["zone_referrer_url"] = value.strip()
        elif normalized == "hosturl":
            result["zone_host_url"] = value.strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(args.manifest)
    output = []
    started = time.perf_counter()
    for index, manifest in enumerate(rows, 1):
        path = Path(manifest["image_path"])
        stat = path.stat()
        parsed = parse_jpeg(path)
        zone = parse_zone_identifier(path)
        available = []
        for label, keys in {
            "EXIF": ["camera_make", "camera_model", "datetime_original", "orientation", "software", "gps_present"],
            "XMP": ["xmp_create_date", "xmp_creator_tool", "xmp_camera_make", "xmp_camera_model"],
            "IPTC": ["iptc_present"], "ICC": ["icc_profile_present"],
            "JFIF": ["jfif_present"], "COM": ["comment_present"],
        }.items():
            if any(parsed.get(key) not in {False, "", 0, None} for key in keys):
                available.append(label)
        identity_fields = [
            parsed.get("datetime_original"), parsed.get("camera_make"),
            parsed.get("camera_model"), parsed.get("xmp_create_date"),
        ]
        confidence = "NONE" if not any(identity_fields) else "PARTIAL"
        notes = []
        if available == ["JFIF"]:
            notes.append("JFIF container metadata only; no capture identity")
        if zone["zone_identifier_present"]:
            available.append("NTFS_ZONE_IDENTIFIER")
            notes.append("Zone.Identifier is transfer/archive provenance, not capture metadata")
        if not parsed["exif_present"]:
            notes.append("no EXIF APP1 segment")
        if not parsed["xmp_present"]:
            notes.append("no XMP APP1 segment")
        output.append(
            {
                "relative_path": path.resolve().relative_to(args.dataset_root.resolve()).as_posix(),
                "file_name": manifest.get("file_name", path.name),
                "original_class": manifest.get("original_class", ""),
                "binary_label": manifest.get("binary_label", ""),
                "split": manifest.get("split", ""),
                "file_size": stat.st_size,
                "windows_creation_time": local_iso(stat.st_ctime),
                "last_write_time": local_iso(stat.st_mtime),
                "extension": path.suffix.lower(),
                **parsed,
                **zone,
                "metadata_fields_available": "|".join(available),
                "capture_identity_confidence": confidence,
                "notes": "; ".join(notes),
            }
        )
        if index % 1000 == 0 or index == len(rows):
            print(f"[metadata] {index}/{len(rows)}", flush=True)
    with (args.output_dir / "metadata_inventory.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)

    def count_true(field: str) -> int:
        return sum(bool(row.get(field)) for row in output)

    summary = {
        "scope": [str(args.dataset_root.resolve()), str(args.manifest.resolve())],
        "files_scanned": len(output),
        "duration_seconds": time.perf_counter() - started,
        "jpeg_valid": count_true("jpeg_valid"),
        "jfif_present": count_true("jfif_present"),
        "exif_present": count_true("exif_present"),
        "iptc_present": count_true("iptc_present"),
        "xmp_present": count_true("xmp_present"),
        "extended_xmp_present": count_true("extended_xmp_present"),
        "icc_profile_present": count_true("icc_profile_present"),
        "photoshop_app13_present": count_true("photoshop_app13_present"),
        "adobe_app14_present": count_true("adobe_app14_present"),
        "comment_present": count_true("comment_present"),
        "camera_make_present": sum(bool(row.get("camera_make") or row.get("xmp_camera_make")) for row in output),
        "camera_model_present": sum(bool(row.get("camera_model") or row.get("xmp_camera_model")) for row in output),
        "capture_time_present": sum(bool(row.get("datetime_original") or row.get("xmp_create_date")) for row in output),
        "gps_present": count_true("gps_present"),
        "orientation_present": sum(row.get("orientation") not in {"", None} for row in output),
        "software_present": sum(bool(row.get("software") or row.get("xmp_creator_tool")) for row in output),
        "zone_identifier_present": count_true("zone_identifier_present"),
        "zone_ids": dict(Counter(str(row["zone_id"]) for row in output).most_common()),
        "zone_referrer_urls": dict(Counter(str(row["zone_referrer_url"]) for row in output).most_common()),
        "zone_host_urls": dict(Counter(str(row["zone_host_url"]) for row in output).most_common()),
        "marker_signatures": dict(Counter(row["marker_signature"] for row in output).most_common()),
        "jfif_versions": dict(Counter(str(row["jfif_version"]) for row in output).most_common()),
        "jfif_density": dict(Counter(
            f"unit={row['jfif_density_units']},x={row['jfif_x_density']},y={row['jfif_y_density']}"
            for row in output
        ).most_common()),
        "creation_time_min": min(row["windows_creation_time"] for row in output),
        "creation_time_max": max(row["windows_creation_time"] for row in output),
        "last_write_time_min": min(row["last_write_time"] for row in output),
        "last_write_time_max": max(row["last_write_time"] for row in output),
        "creation_date_counts": dict(Counter(row["windows_creation_time"][:10] for row in output).most_common()),
        "last_write_date_counts": dict(Counter(row["last_write_time"][:10] for row in output).most_common()),
        "resolution_counts": dict(Counter(
            f"{row['jpeg_width']}x{row['jpeg_height']}" for row in output
        ).most_common()),
        "class_counts": dict(Counter(row["original_class"] for row in output).most_common()),
    }
    (args.output_dir / "metadata_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
