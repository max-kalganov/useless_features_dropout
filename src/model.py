"""Base model implementation"""
import os
from typing import Callable, Optional

import gin
import tensorflow as tf

from src.constants import FEATURES_NUMBER, COMMON_DTYPE, MODEL_CHECKPOINTS_DIR


@gin.configurable
def get_model(
        add_exp_layer: Optional[Callable],
        optimizer,
        loss,
        metrics,
        seed,
        load_model_checkpoint: Optional[str]
) -> tf.keras.models.Model:
    tf.random.set_seed(seed)
    x_input = tf.keras.layers.Input((FEATURES_NUMBER,), dtype=COMMON_DTYPE)

    if load_model_checkpoint is None:
        if add_exp_layer is not None:
            x = add_exp_layer(x_input)
        else:
            x = x_input

        x = tf.keras.layers.Dense(3, activation='relu')(x)
        x = tf.keras.layers.Dense(4, activation='relu')(x)
        x = tf.keras.layers.Dense(1)(x)
    else:
        load_model_checkpoint = os.path.join(MODEL_CHECKPOINTS_DIR, load_model_checkpoint)
        loaded_model = tf.keras.models.load_model(load_model_checkpoint)
        x = x_input
        for layer in loaded_model.layers[1:]:
            if layer.name != "experts_dropout":
                x = layer(x)

    model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    print(model.summary())
    return model
