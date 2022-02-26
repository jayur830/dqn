import numpy as np

from typing import Any


class Environment:
    def __init__(self, init_state: np.ndarray):
        self._init_state = init_state
        self._reward = 0
        self._state = self._init_state.copy()
        self._done = True

    def state(self) -> Any:
        self._state = self._state_preprocess(self._state.copy())
        return self._state.copy()

    def step(self, action: int):
        pass

    def done(self) -> bool:
        return self._done

    def reset(self):
        self._state = self._init_state.copy()
        self._reward = 0
        self._done = False

    def _state_preprocess(self, state: np.ndarray):
        return state
