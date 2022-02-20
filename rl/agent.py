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
        actions = self.__mask(state, q_value)
        return q_value, actions if len(actions) > 1 else actions[0]

    def train(self, x, y):
        self.__q_model.train_on_batch(x, y)

    def update_target_model(self):
        self.__target_model.set_weights(self.__q_model.get_weights())

    def __mask(self, states, q_values):
        actions = np.asarray([0])
        for i in range(states.shape[0]):
            state, q_value = states[i], q_values[i]
            indexes = np.where(state.reshape((state.shape[0] * state.shape[1],)) == 0)[0].reshape(-1)
            actions = np.append(actions, np.where(q_value == np.max(q_value.reshape(-1)[indexes]))[0][0])
        return actions[1:]
