import tensorflow as tf
import numpy as np

from typing import Union, Callable
from rl.agent import Agent


class TicTacToeAgent(Agent):
    def __init__(self, model: tf.keras.models.Model, e_greedy_fn: Callable[[float], float]):
        super().__init__(model, e_greedy_fn)

    def _mask(self, states, q_values) -> Union[np.ndarray, None]:
        flatten_states = states.reshape((states.shape[0], states.shape[1] * states.shape[2]))
        i, j = np.where(flatten_states != 0)
        q_values[i, j] = -10000
        return q_values
