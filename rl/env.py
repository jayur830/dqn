import numpy as np


class Environment:
    def __init__(self, init_state: np.ndarray):
        self._init_state = init_state
        self._state = init_state.copy()

    def step(self, action: int):
        pass

    def reset(self):
        pass
