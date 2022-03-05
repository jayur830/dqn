import numpy as np

from typing import Union, Any


class Environment:
    def __init__(self, init_state: np.ndarray):
        self._init_state = init_state
        self._state = init_state.copy()

    def step(self, action: int) -> Union[np.ndarray, int, bool, Any] :
        pass

    def reset(self) -> np.ndarray:
        pass
