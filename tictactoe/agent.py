import numpy as np

from rl.agent import Agent


class TicTacToeAgent(Agent):
    def _mask(self, states, q_values):
        flatten_states = states.reshape((states.shape[0], states.shape[1] * states.shape[2]))
        i, j = np.where(flatten_states != 0)
        q_values[i, j] = np.min(q_values) - 1
        return q_values.argmax(axis=1)
