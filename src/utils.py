"""Utils"""
import os
from typing import List

from src.dataset_generator import get_dataset
from src.model import get_model
import keras
import pandas as pd
from src import constants as ct


def get_single_experiment_results(
        batch_size: int,
        epochs: int,
        tensorboard_logs: str,
        seed: int
) -> List[float]:
    x_train, y_train, x_test, y_test = get_dataset(seed=seed)
    model = get_model(seed=seed)

    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test, y_test],
              callbacks=[
                  keras.callbacks.TensorBoard(log_dir=tensorboard_logs)
              ])

    return model.evaluate(x_test, y_test)


def dump_results(model_results: List[float], results_file: str, exp_name: str, exp_comments: str):
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    df = pd.read_csv(results_file) if os.path.exists(results_file) else pd.DataFrame()

    new_record = pd.Series({
        ct.EXPERIMENT_NAME: exp_name,
        ct.EXPERIMENT_LOSS: model_results[0],
        ct.EXPERIMENT_ACCURACY: model_results[1],
        ct.EXPERIMENT_COMMENTS: exp_comments
    })

    df = df.append(new_record, ignore_index=True)
    df.to_csv(results_file, index=False)
