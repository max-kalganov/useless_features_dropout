"""Runs experiment using gin config"""
import gin
from src.experiments_runner import ExperimentsRunner
import tensorflow as tf
gin.external_configurable(tf.keras.losses.CategoricalCrossentropy)


if __name__ == '__main__':
    gin.parse_config_file("configs/mnist_perm_feature_importance_exp.gin")
    exp_runner = ExperimentsRunner()
    exp_runner.run_experiment()
