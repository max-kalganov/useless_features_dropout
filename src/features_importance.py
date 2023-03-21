import os
from functools import partial
from typing import Optional, Tuple

import gin
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.inspection import permutation_importance

from src.constants import MODEL_CHECKPOINTS_DIR


class FeatureSelectionMethods:
    def __init__(self, method: str):
        names_to_methods = {
            'all': lambda mean, std: np.arange(0, len(mean)),
            'top_1': partial(self._select_by_num, num_of_top_imp_features_to_leave=1),
            'top_3': partial(self._select_by_num, num_of_top_imp_features_to_leave=3),
            'by_stats': self._select_by_stats
        }
        self.selected_method = names_to_methods[method]

    def __call__(self, feature_importance_mean: np.ndarray, feature_importance_std: np.ndarray) -> np.ndarray:
        return self.selected_method(feature_importance_mean, feature_importance_std)

    @staticmethod
    def _select_by_num(feature_importance_mean: np.ndarray, feature_importance_std: np.ndarray,
                       num_of_top_imp_features_to_leave: int):
        return feature_importance_mean.argsort()[::-1][:num_of_top_imp_features_to_leave]

    @staticmethod
    def _select_by_stats(feature_importance_mean: np.ndarray, feature_importance_std: np.ndarray):
        features_filter = feature_importance_mean - 2 * feature_importance_std
        return np.argwhere(features_filter > 0)


@gin.configurable
def get_feature_importance(model, x, y, n_repeats=30, seed=0) -> Tuple[np.ndarray, np.ndarray]:
    r = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=seed, scoring='r2')
    return r.importances_mean, r.importances_std


@gin.configurable()
def permutation_fi_scores(dumped_model_name: str,
                          ds_train,
                          feature_selection_method: str):
    """Setting up dropout based on permutation feature importance"""

    def load_model() -> tf.keras.models.Model:
        load_model_checkpoint = os.path.join(MODEL_CHECKPOINTS_DIR, dumped_model_name)
        loaded_model = tf.keras.models.load_model(load_model_checkpoint)
        return loaded_model

    def aggregate_data() -> Tuple[np.ndarray, np.ndarray]:
        x_arrays, labels = [], []
        for x, y in tfds.as_numpy(ds_train):
            x_arrays.append(x)
            labels.append(y)

        x = np.vstack(x_arrays)
        y = np.vstack(labels)
        return x, y

    def get_features_scores(feature_importance: np.ndarray, args_to_leave: np.ndarray) -> np.ndarray:
        result_feature_importance = np.zeros(feature_importance.shape)
        result_feature_importance[args_to_leave] = feature_importance[args_to_leave]
        print(f"Result feature scores: {result_feature_importance}")
        return result_feature_importance

    x, y = aggregate_data()
    feature_selection = FeatureSelectionMethods(feature_selection_method)
    fi_mean, fi_std = get_feature_importance(
        model=load_model(),
        x=x,
        y=y
    )
    print(f"Feature importance: {fi_mean}, {fi_std}")
    args_to_leave = feature_selection(fi_mean, fi_std)
    print(f"Args to leave: {args_to_leave}")

    return get_features_scores(fi_mean, args_to_leave=args_to_leave)
