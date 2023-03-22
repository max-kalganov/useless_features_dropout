"""Implementations of custom and regular dropouts to be used on model"""
import sys
from typing import List, Union

import gin
import numpy as np
import tensorflow as tf

from src.constants import COMMON_DTYPE
from src.utils import get_feature_importance_files


class ExpertsDropout(tf.keras.layers.Layer):
    """Drops out input values individually by features scores for each input feature.

    `DrouOut feature i with the prob {feature_score[i]}`

    feature_score == 0 -> feature is always passed forward
    feature_score == 1 -> feature is almost never passed forward (replaced with 0)
    """

    def __init__(self, features_scores: Union[List[float], np.ndarray]):
        super().__init__()
        assert all(0 <= imp <= 1 for imp in features_scores), "incorrect feature importance is set"
        assert isinstance(features_scores, list) or isinstance(features_scores, np.ndarray),\
            "incorrect features_scores type"
        self.features_scores = features_scores

    @tf.function
    def call(self, inputs, training=False):
        result = inputs
        if training:
            prob = tf.random.uniform(shape=(inputs.shape[1], ), minval=0, maxval=1, dtype=COMMON_DTYPE)
            result = tf.where(self.features_scores * 10 <= prob, inputs, 0.)
        return result


@gin.configurable()
def regular_dropout(x_input, prob):
    return tf.keras.layers.Dropout(prob)(x_input)


@gin.configurable()
def experts_dropout(x_input, features_scores):
    # make features score to be 1 if it has to be dropped out
    features_dropout_probs = 1. - np.array(features_scores)
    return ExpertsDropout(features_dropout_probs)(x_input)


@gin.configurable
def feature_importance_dropout(x_input, features_importance_dir: str) -> ExpertsDropout:
    features_importance_file, _ = get_feature_importance_files(features_importance_dir)
    features_importance = np.load(features_importance_file)
    return ExpertsDropout(features_importance)(x_input)
