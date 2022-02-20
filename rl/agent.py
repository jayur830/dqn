import numpy as np
import tensorflow as tf


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
        q_value = np.asarray(self.__target_model(state)).copy()
        masked_result = self._mask(state, q_value)
        actions = q_value.argmax(axis=1) if masked_result is None else masked_result
        return q_value, actions if len(actions) > 1 else actions[0]

    def train(self, x, y):
        self.__q_model.train_on_batch(x, y)

    def update_target_model(self):
        self.__target_model.set_weights(self.__q_model.get_weights())

    def _mask(self, states, q_values):
        return None
