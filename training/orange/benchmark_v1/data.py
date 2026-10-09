"""TensorFlow loaders accepting verifier-issued partitions only."""

from __future__ import annotations

import os

from .context import (
    VerifiedConfig,
    VerifiedTrainDataset,
    VerifiedTrainPartition,
    VerifiedValidationDataset,
    VerifiedValidationPartition,
    _DATASET_SEAL,
)
from .manifest import resolve_verified_paths


def _dataset_root(config: VerifiedConfig) -> str:
    env_name = str(config.values["data"]["dataset_root_env"])
    value = os.environ.get(env_name)
    if not value:
        raise RuntimeError(f"BLOCKED: required environment variable {env_name} is missing.")
    return value


def _build_tf_dataset(config: VerifiedConfig, partition, *, training: bool):
    expected_type = VerifiedTrainPartition if training else VerifiedValidationPartition
    if not isinstance(config, VerifiedConfig) or not isinstance(partition, expected_type):
        raise TypeError("Loader requires verified config and the matching verified partition.")
    paths = resolve_verified_paths(partition, _dataset_root(config))
    labels = [row.label_id for row in partition.records]
    batch_size = int(config.values["data"]["batch_size"])
    image_size = int(config.values["data"]["image_size"])
    seed = int(config.values["project"]["seed"])
    augmentation_config = config.values["augmentation"]

    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(([str(path) for path in paths], labels))
    options = tf.data.Options()
    # Seeded deterministic ordering is requested. Bitwise identity across
    # TensorFlow/CUDA versions and devices is not claimed.
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    def decode(path, label):
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        tf.debugging.assert_equal(tf.shape(image), [image_size, image_size, 3])
        image = tf.cast(image, tf.float32)  # Embedded MobileNetV3 preprocessing expects 0..255.
        return image, tf.reshape(tf.cast(label, tf.float32), [1])

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(min(len(paths), 4096), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip(
                    "horizontal" if augmentation_config["horizontal_flip"] else None,
                    seed=seed,
                ),
                tf.keras.layers.RandomRotation(
                    float(augmentation_config["rotation_factor"]), fill_mode="reflect", seed=seed + 1
                ),
                tf.keras.layers.RandomZoom(
                    float(augmentation_config["zoom_factor"]), fill_mode="reflect", seed=seed + 2
                ),
                tf.keras.layers.RandomTranslation(
                    float(augmentation_config["translation_factor"]),
                    float(augmentation_config["translation_factor"]),
                    fill_mode="reflect", seed=seed + 3,
                ),
                tf.keras.layers.RandomContrast(
                    float(augmentation_config["contrast_factor"]), seed=seed + 4
                ),
            ],
            name="benchmark_v1_train_augmentation",
        )
        dataset = dataset.map(
            lambda images, targets: (augmentation(images, training=True), targets),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE), batch_size


def build_train_dataset(config: VerifiedConfig, partition: VerifiedTrainPartition) -> VerifiedTrainDataset:
    dataset, batch_size = _build_tf_dataset(config, partition, training=True)
    return VerifiedTrainDataset(
        dataset=dataset,
        manifest_sha256=partition.manifest_sha256,
        sample_count=len(partition.records),
        batch_size=batch_size,
        sampling_strategy="NONE",
        config_sha256=config.sha256,
        image_size=int(config.values["data"]["image_size"]),
        _seal=_DATASET_SEAL,
    )


def build_validation_dataset(
    config: VerifiedConfig, partition: VerifiedValidationPartition
) -> VerifiedValidationDataset:
    dataset, batch_size = _build_tf_dataset(config, partition, training=False)
    return VerifiedValidationDataset(
        dataset=dataset,
        manifest_sha256=partition.manifest_sha256,
        sample_count=len(partition.records),
        batch_size=batch_size,
        config_sha256=config.sha256,
        image_size=int(config.values["data"]["image_size"]),
        _seal=_DATASET_SEAL,
    )
