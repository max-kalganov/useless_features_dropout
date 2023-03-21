import os

import gin

from src import get_mnist_dataset
from src.features_importance import permutation_fi_scores


@gin.configurable
def dump_features_importance(features_importance_file: str):
    ds_train, _ = get_mnist_dataset()
    features_importance = permutation_fi_scores(ds_train=ds_train)
    os.makedirs(os.path.dirname(features_importance_file), exist_ok=True)
    features_importance.save(features_importance_file)
    print(f"Features importance {features_importance} are saved into {features_importance_file}")


if __name__ == '__main__':
    gin.parse_config_file("configs/features_importance_config.gin")
    dump_features_importance()
