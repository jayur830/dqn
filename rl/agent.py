import tensorflow as tf
import numpy as np

from typing import Callable


class Agent:
    def __init__(self, model: tf.keras.models.Model):
        self.__target_model = model
        self.__q_model = tf.keras.models.clone_model(model)
        self.__q_model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics)
        self.__q_model.set_weights(model.get_weights())

    def predict(self, state):
        if len(state.shape) < 3:
            state = state.reshape((1,) + state.shape)
        q_value = self.__target_model.predict(state)
        action = q_value.reshape(-1).argmax()
        return q_value, action

    def train(self, x, y):
        self.__q_model.train_on_batch(x, y)

    def update_target_model(self):
        self.__target_model.set_weights(self.__q_model.get_weights())
