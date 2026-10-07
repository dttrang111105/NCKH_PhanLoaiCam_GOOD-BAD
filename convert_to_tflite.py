from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


APP_DIR = Path(__file__).resolve().parent
IMAGE_SIZE = 224


def representative_dataset(dataset_dir: Path, max_images: int = 300):
    extensions = {".jpg", ".jpeg", ".png"}
    image_paths = [
        p for p in dataset_dir.rglob("*")
        if p.suffix.lower() in extensions
    ]

    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh representative trong: {dataset_dir}")

    rng = np.random.default_rng(42)
    if len(image_paths) > max_images:
        image_paths = list(rng.choice(image_paths, max_images, replace=False))

    print(f"[INT8] Representative dataset: {len(image_paths)} ảnh")

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            image = np.asarray(image, dtype=np.float32)
            image = np.expand_dims(image, axis=0)
            yield [image]
        except Exception:
            continue


def convert(
    keras_model_path: Path,
    threshold_json_path: Path,
    dataset_dir: Path,
    max_images: int = 300,
) -> None:
    import tensorflow as tf

    if not keras_model_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model đã train: {keras_model_path}"
        )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset representative: {dataset_dir}"
        )

    print("=" * 60)
    print("CONVERT MOBILEV3-SMALL -> TFLITE INT8")
    print("=" * 60)
    print(f"Model:   {keras_model_path}")
    print(f"Dataset: {dataset_dir}")

    print("\n[1/5] Load Keras model...")
    model = tf.keras.models.load_model(
        keras_model_path,
        compile=False,
    )

    saved_model_dir = Path(
        tempfile.mkdtemp(prefix="orange_saved_model_")
    )

    try:
        print("\n[2/5] Export SavedModel...")
        model.export(saved_model_dir)

        print("\n[3/5] Convert sang INT8...")
        converter = tf.lite.TFLiteConverter.from_saved_model(
            str(saved_model_dir)
        )

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.representative_dataset = lambda: representative_dataset(
            dataset_dir,
            max_images,
        )

        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]

        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        output_path = APP_DIR / "model.tflite"
        output_path.write_bytes(tflite_model)

        size_mb = output_path.stat().st_size / (1024 * 1024)

        print(f"\n[4/5] Đã tạo: {output_path}")
        print(f"Kích thước: {size_mb:.2f} MB")

        print("\n[5/5] Kiểm tra INT8...")
        interpreter = tf.lite.Interpreter(
            model_path=str(output_path)
        )
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_info = input_details[0]
        output_info = output_details[0]

        print(f"Input shape : {input_info['shape']}")
        print(f"Input dtype : {input_info['dtype']}")
        print(f"Input scale : {input_info['quantization'][0]}")
        print(f"Input zero  : {input_info['quantization'][1]}")

        print(f"Output shape: {output_info['shape']}")
        print(f"Output dtype: {output_info['dtype']}")
        print(f"Output scale: {output_info['quantization'][0]}")
        print(f"Output zero : {output_info['quantization'][1]}")

        if input_info["dtype"] != np.int8:
            raise RuntimeError(
                f"Input chưa phải INT8: {input_info['dtype']}"
            )

        if output_info["dtype"] != np.int8:
            raise RuntimeError(
                f"Output chưa phải INT8: {output_info['dtype']}"
            )

        if input_info["quantization"][0] == 0:
            raise RuntimeError("Input INT8 chưa có quantization scale.")

        if output_info["quantization"][0] == 0:
            raise RuntimeError("Output INT8 chưa có quantization scale.")

        print("\n[OK] Input = INT8")
        print("[OK] Output = INT8")
        print("[OK] Model đã được lượng tử hóa INT8 hoàn toàn.")

        if threshold_json_path.exists():
            payload = json.loads(
                threshold_json_path.read_text(
                    encoding="utf-8"
                )
            )

            threshold = float(
                payload.get(
                    "final_threshold",
                    payload.get("threshold", 0.14),
                )
            )

            threshold_path = APP_DIR / "threshold.json"

            threshold_path.write_text(
                json.dumps(
                    {"threshold": threshold},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(
                f"[OK] threshold.json = {threshold:.4f}"
            )
        else:
            print(
                f"[Cảnh báo] Không tìm thấy threshold: "
                f"{threshold_json_path}"
            )

        print("\n" + "=" * 60)
        print("CONVERT INT8 HOÀN TẤT")
        print("=" * 60)
        print(f"Model   : {output_path}")
        print(f"Size    : {size_mb:.2f} MB")
        print("Input   : INT8")
        print("Output  : INT8")
        print("=" * 60)

    finally:
        shutil.rmtree(
            saved_model_dir,
            ignore_errors=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MobileNetV3-Small sang TFLite INT8."
    )

    parser.add_argument(
        "--keras-model",
        type=str,
        default="training/outputs/models/best_model.keras",
        help="Model Keras đã train.",
    )

    parser.add_argument(
        "--threshold-json",
        type=str,
        default="training/outputs/selected_threshold.json",
        help="File threshold đã chọn.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="E:/dataset",
        help="Dataset dùng làm representative dataset.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=300,
        help="Số ảnh representative dùng để calibration.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        convert(
            keras_model_path=Path(
                args.keras_model
            ).expanduser().resolve(),
            threshold_json_path=Path(
                args.threshold_json
            ).expanduser().resolve(),
            dataset_dir=Path(
                args.dataset
            ).expanduser().resolve(),
            max_images=args.max_images,
        )
        return 0

    except KeyboardInterrupt:
        print("\nĐã dừng.")
        return 130

    except Exception as exc:
        print(f"\n[LỖI] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
