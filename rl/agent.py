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

    def predict(self, state, training=True):
        q_value = self.__target_model.predict(state)
        # if not training:
        indexes = np.transpose(np.where(state.reshape((state.shape[0], state.shape[1] * state.shape[2], state.shape[3])) != 0))
        for i in range(indexes.shape[0]):
            q_value[indexes[i][0], indexes[i][1]] = -999
        action = q_value.reshape(-1).argmax()
        return q_value, action

    def train(self, x, y):
        self.__q_model.train_on_batch(x, y)

    def update_target_model(self):
        self.__target_model.set_weights(self.__q_model.get_weights())
