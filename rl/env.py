import numpy as np


class Environment:
    def __init__(self, init_state: np.ndarray):
        self._init_state = init_state
        self._reward = 0
        self._state = self._init_state.copy()
        self._done = True

    def state(self):
        return self._state.copy()

    def step(self, action):
        pass

    def done(self):
        return self._done

    def reset(self):
        self._state = self._init_state.copy()
        self._done = False
