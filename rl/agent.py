import tensorflow as tf
import numpy as np
import random

from typing import Callable, Union


class Agent:
    def __init__(self, model: tf.keras.models.Model, e_greedy_fn: Callable[[float], float], epsilon: float = .9):
        self.__target_model = model
        self.__q_model = tf.keras.models.clone_model(model)
        self.__q_model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics)
        self.__q_model.set_weights(model.get_weights())
        self.__e_greedy_fn = e_greedy_fn
        self.__epsilon = epsilon
        self.learning_rate = model.optimizer.learning_rate

    def predict(self, state):
        q_value = np.asarray(self.__target_model(state)).copy()
        masked_q_value = self._mask(state, q_value)
        self.__epsilon = self.__e_greedy_fn(self.__epsilon)
        if random.random() < self.__epsilon:
            indexes = np.where(masked_q_value.reshape(-1) != np.min(masked_q_value))[0]
            actions = masked_q_value.argmax(axis=1) if indexes.shape[0] == 0 else indexes[np.random.randint(indexes.shape[0])]
        else:
            actions = masked_q_value.argmax(axis=1)
        try:
            actions = actions if actions.shape[0] > 1 else actions[0]
        except IndexError:
            pass
        return q_value, actions

    def train(self, x, y):
        self.__q_model.train_on_batch(x, y)

    def update_target_model(self):
        self.__target_model.set_weights(self.__q_model.get_weights())

    def save(self, save_path: str = "model.h5"):
        self.__target_model.save(filepath=save_path)

    def _mask(self, states, q_values) -> Union[np.ndarray, None]:
        return q_values
