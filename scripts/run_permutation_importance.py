import os

import gin
import numpy as np

from src import get_heart_attack_dataset
from src.features_importance import permutation_fi_scores
from src.utils import get_feature_importance_files


@gin.configurable
def dump_features_importance(features_importance_dir: str):
    features_importance_file, args_to_leave_file = get_feature_importance_files(features_importance_dir)

    ds_train, _ = get_heart_attack_dataset()
    features_importance, args_to_leave = permutation_fi_scores(ds_train=ds_train)
    os.makedirs(os.path.dirname(features_importance_file), exist_ok=True)
    np.save(features_importance_file, features_importance)
    np.save(args_to_leave_file, args_to_leave)
    print(f"Features importance {features_importance} are saved into {features_importance_file}")
    print(f"Args to leave {args_to_leave} are saved into {args_to_leave_file}")


if __name__ == '__main__':
    gin.parse_config_file("configs/features_importance_config.gin")
    dump_features_importance()
