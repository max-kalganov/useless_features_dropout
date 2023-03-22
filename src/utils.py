import os
from typing import Tuple


def get_feature_importance_files(features_importance_dir: str) -> Tuple[str, str]:
    feature_importance_file = os.path.join(features_importance_dir, 'feature_importance.np')
    args_to_leave_file = os.path.join(features_importance_dir, 'args_to_leave.np')
    return feature_importance_file, args_to_leave_file
