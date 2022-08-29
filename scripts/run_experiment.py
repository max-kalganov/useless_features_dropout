"""Runs experiment using gin config"""
import os.path
from typing import Optional

import gin

from src.constants import TENSORBOARD_LOGS_DIR, MODEL_CHECKPOINTS_DIR
from src.utils import dump_results, get_single_experiment_results


@gin.configurable()
def run_experiment(
        batch_size: int,
        epochs: int,
        tensorboard_logs_name: str,
        results_file: str,
        exp_name: str,
        exp_comments: str,
        save_model_checkpoint: Optional[str]
) -> None:
    tensorboard_logs = os.path.join(TENSORBOARD_LOGS_DIR, tensorboard_logs_name)
    save_model_path = os.path.join(MODEL_CHECKPOINTS_DIR, save_model_checkpoint) if save_model_checkpoint is not None else None

    exp_results = get_single_experiment_results(batch_size,
                                                epochs,
                                                tensorboard_logs,
                                                seed=0,
                                                save_model_checkpoint=save_model_path)
    dump_results(exp_results, results_file, exp_name, exp_comments)


if __name__ == '__main__':
    gin.parse_config_file("configs/config.gin")

    run_experiment()
