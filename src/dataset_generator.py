"""Dataset generator implementation"""
from typing import Tuple

import gin
import numpy as np

from src.constants import FEATURES_NUMBER
import tensorflow_datasets as tfds
import tensorflow as tf


@gin.configurable
def get_dataset(
        n_samples: int,
        train_test_split: float,
        seed: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    np.random.seed(seed)
    x = np.random.random(size=(n_samples, FEATURES_NUMBER))
    y = 10 * x[:, 0] * (x[:, 0] + 5 * x[:, 1]) - 1/100 * x[:, 2]

    split = int(n_samples * train_test_split)

    train_dataset = tf.data.Dataset.from_tensor_slices((x[:split, :], y[:split]))
    test_dataset = tf.data.Dataset.from_tensor_slices((x[split:, :], y[split:]))
    return train_dataset, test_dataset


@gin.configurable
def get_mnist_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    def reshape_img(image, label):
        """Flatten image"""
        return tf.reshape(image, (-1,)), label

    def process_dataset(dataset, is_train: bool):
        dataset = dataset.map(
            normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            reshape_img, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.cache()
        if is_train:
            dataset = dataset.shuffle(ds_info.splits['train'].num_examples)
        dataset = dataset.batch(128)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = process_dataset(ds_train, is_train=True)
    ds_test = process_dataset(ds_test, is_train=False)
    return ds_train, ds_test
