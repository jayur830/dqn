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

    def predict(self, state):
        q_value = np.asarray(self.__target_model(state)).copy()
        masked_result = self._mask(state, q_value)
        argmax_values = q_value if masked_result is None else masked_result
        self.__epsilon = self.__e_greedy_fn(self.__epsilon)
        if random.random() < self.__epsilon:
            actions = np.random.randint(q_value.shape[-1], size=q_value.shape[0])
        else:
            actions = argmax_values.argmax(axis=1)
        return q_value, actions if len(actions) > 1 else actions[0]

    def train(self, x, y):
        self.__q_model.train_on_batch(x, y)

    def update_target_model(self):
        self.__target_model.set_weights(self.__q_model.get_weights())

    def _mask(self, states, q_values) -> Union[np.ndarray, None]:
        return None
