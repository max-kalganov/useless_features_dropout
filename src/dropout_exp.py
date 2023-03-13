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
def get_feature_importance(model, x, y, n_repeats=30, seed=0):
    r = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=seed)
    return r.importances_mean


@gin.configurable()
def permutation_fi_dropout(x_input,
                           dumped_model_name: str,
                           ds_train,
                           num_of_top_imp_features_to_leave: Optional[int] = None):
    """Setting up dropout based on permutation feature importance"""
    def load_model():
        load_model_checkpoint = os.path.join(MODEL_CHECKPOINTS_DIR, dumped_model_name)
        loaded_model = tf.keras.models.load_model(load_model_checkpoint)
        return loaded_model

    def aggregate_data():
        array = np.vstack(tfds.as_numpy(ds_train[0]))
        x_train = np.array(list(map(lambda x: x[0], array)))
        y_train = np.array(list(map(lambda x: x[1], array)))
        return x_train, y_train

    def get_features_scores(feature_importance):
        if num_of_top_imp_features_to_leave:
            highest_score_to_select = feature_importance.argsort()[::-1][num_of_top_imp_features_to_leave - 1]
        else:
            highest_score_to_select = 0
        feature_importance[feature_importance < highest_score_to_select] = 0
        feature_importance = 1 - feature_importance
        return feature_importance

    x, y = aggregate_data()
    fi = get_feature_importance(
        model=load_model(),
        x=x,
        y=y
    )

    return ExpertsDropout(get_features_scores(fi))(x_input)
