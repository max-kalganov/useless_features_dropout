"""Base model implementation"""
from typing import Callable, Optional

import gin
import tensorflow as tf

from constants import FEATURES_NUMBER


@gin.configurable
def get_model(add_exp_layer: Optional[Callable], optimizer, loss, metrics) -> tf.keras.models.Model:
    x_input = tf.keras.layers.Input((FEATURES_NUMBER,))

    if add_exp_layer is not None:
        x = add_exp_layer(x_input)
    else:
        x = x_input

    x = tf.keras.layers.Dense(3, activation='relu')(x)
    x = tf.keras.layers.Dense(4, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=[x_input], outputs=[x])
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
