"""Implementations of custom and regular dropouts to be used on model"""
import os
from typing import List, Union, Optional

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.inspection import permutation_importance

from src.constants import COMMON_DTYPE, MODEL_CHECKPOINTS_DIR


class ExpertsDropout(tf.keras.layers.Layer):
    def __init__(self, features_scores: Union[List[float], np.ndarray]):
        super().__init__()
        assert all(0 <= imp <= 1 for imp in features_scores), "incorrect feature importance is set"
        assert isinstance(features_scores, list), "incorrect features_scores type"
        self.features_scores = features_scores

    def call(self, inputs, training=False):
        result = inputs
        if training and inputs.shape[0] is not None:
            prob = tf.random.uniform(shape=inputs.shape, minval=0, maxval=1, dtype=COMMON_DTYPE)
            result = tf.where(prob <= self.features_scores, inputs, 0.)
        return result


@gin.configurable()
def regular_dropout(x_input, prob):
    return tf.keras.layers.Dropout(prob)(x_input)


@gin.configurable()
def experts_dropout(x_input, features_scores):
    return ExpertsDropout(features_scores)(x_input)


@gin.configurable
def feature_importance_dropout(x_input, features_importance_file: str) -> ExpertsDropout:
    features_importance = np.load(features_importance_file)
    return ExpertsDropout(features_importance)(x_input)
