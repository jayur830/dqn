import numpy as np
import random

from rl.env import Environment

indices = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    [1, 1],
    [1, 2],
    [2, 0],
    [2, 1],
    [2, 2]
]


class TicTacToeEnvironment(Environment):
    def step(self, action):
        action = int(action)
        next_state = self._state.copy()
        random_indexes = np.where(self._state.reshape(-1) == 0)[0]
        random_indexes = random_indexes[np.where(random_indexes != action)[0]]
        if len(random_indexes) == 0:
            self._done = True
            self._reward = -1
            return self._reward, next_state
        random_index = random_indexes[random.randint(0, len(random_indexes) - 1)]
        if self.__exist(action):
            self._reward = -100
            self._done = True
        else:
            next_state[indices[action][0], indices[action][1]] = 1.
            if action != random_index:
                next_state[indices[random_index][0], indices[random_index][1]] = -1.
            self._state = next_state
            self._done, winner = self.__result(next_state)
            if winner == -1:
                self._reward = -10
            elif winner == 1:
                self._reward = max(self._reward, 100)
            else:
                self._reward = -1 if self._done else (max(self._reward, 0) + 1) * 2
        return self._reward, next_state

    def __exist(self, action):
        return self._state[indices[action][0], indices[action][1]] != 0

    def __result(self, next_state):
        vertical_sum_state, horizontal_sum_state = next_state.sum(axis=0), next_state.sum(axis=1)
        if -3 in vertical_sum_state or -3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == -3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == -3:
            return True, -1
        elif 3 in vertical_sum_state or 3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == 3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == 3:
            return True, 1
        elif len(np.where(next_state == 0)[0]) == 0:
            return True, 0
        else:
            return False, 0
