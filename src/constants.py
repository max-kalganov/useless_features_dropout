"""Constants"""
import tensorflow as tf


FEATURES_NUMBER = 5
TENSORBOARD_LOGS_DIR = "data/logs"
MODEL_CHECKPOINTS_DIR = "data/checkpoints"
COMMON_DTYPE = tf.float32

# Dump results col names
EXPERIMENT_NAME = "exp_name"
EXPERIMENT_LOSS = "exp_loss"
EXPERIMENT_ACCURACY = "exp_acc"
EXPERIMENT_COMMENTS = "exp_comments"

HEART_ATTACK_DATASET_PATH = "data/heart_attack/heart.csv"
HEART_ATTACK_LABEL = "output"
