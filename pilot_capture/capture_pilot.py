"""Interactive pilot capture with append-only provenance manifest.

The script saves each camera frame at the resolution returned by OpenCV as a
lossless PNG. It never resizes or overwrites an existing image. A manifest row
is appended and fsynced immediately after the image SHA-256 is calculated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2


POLICY_VERSION = "LABEL_POLICY_V1"
CAPTURE_TIME_SOURCE = "HOST_FRAME_RECEIVED"
ORIGINAL_CLASSES = {
    "fresh_orange", "green_orange", "blackspotted_orange",
    "cankered_orange", "rotten_orange",
}
BINARY_LABELS = {"GOOD", "BAD"}
FIELDS = [
    "image_path", "fruit_id", "session_id", "camera_id", "capture_time",
    "capture_time_source", "original_class", "binary_label", "label_reason",
    "view_id", "defect_visible", "review_status", "source_sha256",
    "capture_width", "capture_height", "policy_version", "notes",
]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class FruitDecision:
    fruit_id: str
    original_class: str
    binary_label: str
    label_reason: str
    review_status: str
    notes: str


@dataclass(frozen=True)
class CapturedFrame:
    image: object
    host_received_at: datetime


def require_id(value: str, field_name: str) -> str:
    value = value.strip()
    if not SAFE_ID.fullmatch(value):
        raise ValueError(
            f"{field_name} phải dài 1-64 ký tự và chỉ gồm chữ, số, _, -, ."
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receive_frame(camera, clock=None) -> CapturedFrame:
    """Return a frame bundled with the host time observed after camera.read().

    OpenCV does not expose a reliable hardware capture timestamp for this
    workflow. The stored value is therefore the host time at which read()
    returned, captured before preview, keyboard handling or metadata prompts.
    """
    if clock is None:
        clock = lambda: datetime.now().astimezone()
    ok, frame = camera.read()
    host_received_at = clock()
    if not ok:
        raise RuntimeError("Camera không trả frame")
    return CapturedFrame(image=frame, host_received_at=host_received_at)


def initialize_manifest(path: Path) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
        if header != FIELDS:
            raise ValueError(f"Manifest đã tồn tại nhưng sai schema: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerow(FIELDS)
        handle.flush()
        os.fsync(handle.fileno())


def existing_decisions(path: Path) -> dict[str, FruitDecision]:
    decisions: dict[str, FruitDecision] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            current = FruitDecision(
                fruit_id=row["fruit_id"],
                original_class=row["original_class"],
                binary_label=row["binary_label"],
                label_reason=row["label_reason"],
                review_status=row["review_status"],
                notes=row["notes"],
            )
            previous = decisions.get(current.fruit_id)
            if previous and previous != current:
                raise ValueError(
                    f"Fruit {current.fruit_id} có metadata không nhất quán trong manifest"
                )
            decisions[current.fruit_id] = current
    return decisions


def next_view_number(path: Path, fruit_id: str) -> int:
    numbers: list[int] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["fruit_id"] != fruit_id:
                continue
            view_id = row["view_id"]
            if view_id in seen:
                raise ValueError(f"view_id bị trùng cho {fruit_id}: {view_id}")
            seen.add(view_id)
            if not re.fullmatch(r"V\d{3,}", view_id):
                raise ValueError(f"view_id không hợp lệ cho {fruit_id}: {view_id}")
            numbers.append(int(view_id[1:]))
    return max(numbers, default=0) + 1


def append_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def prompt_choice(prompt: str, choices: set[str], allow_blank: bool = False) -> str:
    while True:
        value = input(prompt).strip()
        if allow_blank and not value:
            return ""
        normalized = value.upper()
        if normalized in choices:
            return normalized
        print(f"Giá trị hợp lệ: {', '.join(sorted(choices))}")


def prompt_required(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Trường này bắt buộc để truy lại lý do quyết định.")


def prompt_fruit(known: dict[str, FruitDecision]) -> FruitDecision:
    while True:
        try:
            fruit_id = require_id(input("fruit_id: "), "fruit_id")
            break
        except ValueError as error:
            print(error)
    if fruit_id in known:
        decision = known[fruit_id]
        print(
            f"Dùng lại {fruit_id}: class={decision.original_class or '<blank>'}, "
            f"label={decision.binary_label or '<blank>'}, status={decision.review_status}"
        )
        return decision

    status = prompt_choice("review_status [APPROVED/REVIEW]: ", {"APPROVED", "REVIEW"})
    print("original_class:", ", ".join(sorted(ORIGINAL_CLASSES)), "hoặc để trống nếu REVIEW")
    original_class = input("original_class: ").strip().lower()

    if status == "REVIEW":
        if original_class and original_class not in ORIGINAL_CLASSES:
            raise ValueError("original_class không thuộc policy; để trống nếu chưa xác định")
        reason = prompt_required("label_reason (lý do REVIEW): ")
        notes = input("notes (có thể để trống): ").strip()
        return FruitDecision(fruit_id, original_class, "", reason, status, notes)

    if original_class not in ORIGINAL_CLASSES:
        raise ValueError("APPROVED bắt buộc original_class hợp lệ")
    if original_class == "green_orange":
        print("green_orange không tự quyết định nhãn: kiểm tra độ chín và chuẩn thương mại của quả thật.")
    label = prompt_choice("binary_label [GOOD/BAD]: ", BINARY_LABELS)
    reason = prompt_required("label_reason (lý do quyết định): ")
    notes = input("notes (có thể để trống): ").strip()
    return FruitDecision(fruit_id, original_class, label, reason, status, notes)


def open_camera(index: int, width: int | None, height: int | None):
    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Không mở được camera index {index}")
    if width:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def capture_fruit(
    camera,
    decision: FruitDecision,
    session_id: str,
    camera_id: str,
    originals_dir: Path,
    manifest_path: Path,
    start_view: int,
) -> int:
    view_number = start_view
    window = "Pilot capture | SPACE=chup | Q=ket thuc qua | ESC=ket thuc session"
    print("SPACE: chụp view; Q: kết thúc quả; ESC: kết thúc cả session")
    while True:
        captured = receive_frame(camera)
        frame = captured.image
        preview = frame.copy()
        cv2.putText(
            preview,
            f"{decision.fruit_id} {decision.binary_label or 'REVIEW'} V{view_number:03d}",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.imshow(window, preview)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return -1
        if key in (ord("q"), ord("Q")):
            return view_number
        if key != 32:
            continue

        view_id = f"V{view_number:03d}"
        defect_visible = "false"
        if decision.binary_label == "BAD":
            defect_visible = prompt_choice(
                "defect_visible [TRUE/FALSE/UNKNOWN]: ",
                {"TRUE", "FALSE", "UNKNOWN"},
            ).lower()
        elif decision.review_status == "REVIEW":
            defect_visible = "unknown"

        captured_at = captured.host_received_at
        stamp = captured_at.strftime("%Y%m%dT%H%M%S_%f%z")
        file_name = f"{session_id}__{decision.fruit_id}__{view_id}__{stamp}.png"
        image_path = originals_dir / file_name
        if image_path.exists():
            raise FileExistsError(f"Từ chối ghi đè: {image_path}")
        if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise OSError(f"Không ghi được ảnh: {image_path}")
        digest = sha256_file(image_path)
        height, width = frame.shape[:2]
        relative_path = image_path.relative_to(manifest_path.parent).as_posix()
        append_row(manifest_path, {
            "image_path": relative_path,
            "fruit_id": decision.fruit_id,
            "session_id": session_id,
            "camera_id": camera_id,
            "capture_time": captured_at.isoformat(timespec="milliseconds"),
            "capture_time_source": CAPTURE_TIME_SOURCE,
            "original_class": decision.original_class,
            "binary_label": decision.binary_label,
            "label_reason": decision.label_reason,
            "view_id": view_id,
            "defect_visible": defect_visible,
            "review_status": decision.review_status,
            "source_sha256": digest,
            "capture_width": width,
            "capture_height": height,
            "policy_version": POLICY_VERSION,
            "notes": decision.notes,
        })
        print(f"Đã lưu {relative_path} | {width}x{height} | sha256={digest}")
        view_number += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Chụp pilot có fruit_id và manifest append-only")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--camera-id", required=True, help="ID vật lý do người chụp đặt, ví dụ CAM01")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--init-only", action="store_true", help="Chỉ tạo session/manifest, không mở camera")
    args = parser.parse_args()

    session_id = require_id(args.session_id, "session_id")
    camera_id = require_id(args.camera_id, "camera_id")
    session_dir = args.output_root.resolve() / session_id
    originals_dir = session_dir / "originals"
    originals_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.csv"
    initialize_manifest(manifest_path)
    print(f"Session: {session_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Originals: {originals_dir}")
    if args.init_only:
        return 0

    camera = open_camera(args.camera_index, args.width, args.height)
    try:
        while True:
            known = existing_decisions(manifest_path)
            try:
                decision = prompt_fruit(known)
            except ValueError as error:
                print(f"Không chấp nhận metadata: {error}")
                continue
            prior_views = next_view_number(manifest_path, decision.fruit_id)
            result = capture_fruit(
                camera, decision, session_id, camera_id,
                originals_dir, manifest_path, prior_views,
            )
            if result == -1:
                break
            again = prompt_choice("Chụp quả khác? [Y/N]: ", {"Y", "N"})
            if again == "N":
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print("Đã kết thúc session. Không resize hoặc chạy model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
