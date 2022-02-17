import tensorflow as tf
import numpy as np

from typing import Callable


class Agent:
    def __init__(self, model: tf.keras.models.Model):
        self.__model = model

    def predict(self, state):
        q_value = np.asarray(self.__model(state))
        action = q_value.argmax()
        return q_value, action

    def action(self, state: np.ndarray, mask: Callable[[np.ndarray, np.ndarray], np.ndarray] = None):
        output = np.asarray(self.__model(state))
        if mask is not None:
            output = mask(state, output)
        print(output)
        # np.asarray(self.call(state, training))
        return output, 0

    def train(self, x, y):
        self.__model.train_on_batch(x, y)