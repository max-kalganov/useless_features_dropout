"""Runs experiment using gin config"""
import gin
import tensorflow as tf
from src.model import get_model
from src.dataset_generator import get_dataset


def dump_results(model_results, results_file: str, exp_name: str):
    pass


@gin.configurable()
def run_experiment(batch_size: int, epochs: int, tensorboard_logs: str, results_file: str, exp_name: str):
    x_train, y_train, x_test, y_test = get_dataset()
    model = get_model()

    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_test, y_test],
              callbacks=[
                  tf.keras.callbacks.Tensorboard(log_dir=tensorboard_logs)
              ])

    dump_results(model.evaluate(x_test, y_test), results_file, exp_name)


if __name__ == '__main__':
    gin.parse_config_file("../configs/config.gin")

    run_experiment()
