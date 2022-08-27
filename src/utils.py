"""Utils"""
import os

from src.dataset_generator import get_dataset
from src.model import get_model
import keras
import pandas as pd


def get_single_experiment_results(
        batch_size: int,
        epochs: int,
        tensorboard_logs: str,
        seed: int
) -> None:
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


def dump_results(model_results, results_file: str, exp_name: str):
    os.makedirs(os.path.dirname(results_file))

    df = pd.read_csv(results_file, index=False) if os.path.exists(results_file) else pd.DataFrame()

    new_record = pd.Series({""})

