"""Implementations of custom and regular dropouts to be used on model"""
from typing import List

import gin
import tensorflow as tf

from src.constants import COMMON_DTYPE


class ExpertsDropout(tf.keras.layers.Layer):
    def __init__(self, features_scores: List[float]):
        super().__init__()
        assert all(0 <= imp <= 1 for imp in features_scores), "incorrect feature importance is set"
        assert isinstance(features_scores, list), "incorrect features_scores type"
        self.features_scores = features_scores

    def call(self, inputs, training=False):
        result = inputs
        if training:
            tf.print("It is training")
            prob = tf.random.uniform(shape=inputs.shape, minval=0, maxval=1, dtype=COMMON_DTYPE)
            result = tf.where(prob < self.features_scores, inputs, 0.)
        return result


@gin.configurable()
def regular_dropout(x_input, prob):
    return tf.keras.layers.Dropout(prob)(x_input)


@gin.configurable()
def experts_dropout(x_input, features_scores):
    return ExpertsDropout(features_scores)(x_input)
