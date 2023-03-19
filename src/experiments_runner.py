import os
from typing import Optional, List, Callable

import gin

from src import constants as ct
import pandas as pd
import tensorflow as tf
import logging
exp_logger = logging.getLogger('Experiment Logger')


@gin.configurable()
class ExperimentsRunner:
    def __init__(
            self,
            get_dataset_callback: Callable,
            get_model_callback: Callable,
            batch_size: int,
            epochs: int,
            steps_per_epoch: int,
            tensorboard_logs_name: str,
            results_file: str,
            exp_name: str,
            exp_comments: str,
            save_model_checkpoint: Optional[str]
    ):
        self.get_dataset_fnc = get_dataset_callback
        self.get_model_fnc = get_model_callback
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.tensorboard_logs_name = tensorboard_logs_name
        self.results_file = results_file
        self.exp_name = exp_name
        self.exp_comments = exp_comments
        self.save_model_checkpoint = save_model_checkpoint

    def run_experiment(self) -> None:
        tensorboard_logs = os.path.join(ct.TENSORBOARD_LOGS_DIR, self.tensorboard_logs_name)
        save_model_path = os.path.join(ct.MODEL_CHECKPOINTS_DIR,
                                       self.save_model_checkpoint) if self.save_model_checkpoint is not None else None

        exp_results = self.get_single_experiment_results(tensorboard_logs,
                                                         seed=0,
                                                         save_model_checkpoint=save_model_path)
        self.dump_results(exp_results)

    def get_single_experiment_results(
            self,
            tensorboard_logs: str,
            seed: int,
            save_model_checkpoint: Optional[str] = None
    ) -> List[float]:
        train_dataset, test_dataset = self.get_dataset_fnc(seed=seed)
        train_dataset = train_dataset.batch(self.batch_size)
        test_dataset = test_dataset.batch(len(test_dataset))
        exp_logger.info(f'Extracted dataset: train - {train_dataset}, test - {test_dataset}')

        model = self.get_model_fnc(seed=seed)
        exp_logger.info(f'Extracted model: {model}')

        model.fit(train_dataset,
                  epochs=self.epochs,
                  steps_per_epoch=self.steps_per_epoch,
                  validation_data=test_dataset,
                  callbacks=[
                      tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logs)
                  ])

        if save_model_checkpoint is not None:
            model.save(save_model_checkpoint)
            exp_logger.info(f'Model is saved into {save_model_checkpoint}')
        return model.evaluate(test_dataset)

    def dump_results(self, model_results: List[float]):
        os.makedirs(os.path.dirname(self.results_file), exist_ok=True)

        df = pd.read_csv(self.results_file) if os.path.exists(self.results_file) else pd.DataFrame()

        new_record = pd.Series({
            ct.EXPERIMENT_NAME: self.exp_name,
            ct.EXPERIMENT_LOSS: model_results[0],
            ct.EXPERIMENT_ACCURACY: model_results[1],
            ct.EXPERIMENT_COMMENTS: self.exp_comments
        })

        df = df.append(new_record, ignore_index=True)
        df.to_csv(self.results_file, index=False)
