import random

import numpy as np

from rl.env import Environment
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win

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
        next_state = self._state.copy()
        index = int(action)
        info = {
            "status": ""
        }

        if not next_state.all():
            self._reward, next_state, info = self.__play(next_state, index, 1.)

        if self._done:
            return self._reward, next_state, info

        if not next_state.all():
            random_indexes = np.where(self._state.reshape(-1) == 0)[0]
            random_indexes = random_indexes[np.where(random_indexes != action)[0]]
            index = random_indexes[random.randint(0, len(random_indexes) - 1)]
            self._reward, next_state, info = self.__play(next_state, index, -1.)

        self._state = next_state
        return self._reward, next_state, info

    def __play(self, next_state, index, target_value):
        info = {
            "status": ""
        }
        if next_state[indices[index][0], indices[index][1]] != 0:
            self._reward = reward_reset
            self._done = True
            info["status"] = "RESET"
            return self._reward, next_state, info
        next_state[indices[index][0], indices[index][1]] = target_value
        self._done, winner, reward = self.__result(next_state)
        self._reward = reward
        if winner == -1:
            info["status"] = "LOSE"
        elif winner == 1:
            info["status"] = "WIN"
        elif winner == 0:
            info["status"] = "DRAW" if self._done else "PLAY"
        return self._reward, next_state, info

    def __result(self, next_state):
        vertical_sum_state, horizontal_sum_state = next_state.sum(axis=0), next_state.sum(axis=1)
        if -3 in vertical_sum_state or -3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == -3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == -3:
            return True, -1, reward_lose
        elif 3 in vertical_sum_state or 3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == 3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == 3:
            return True, 1, reward_win
        elif len(np.where(next_state == 0)[0]) == 0:
            return True, 0, reward_draw
        else:
            return False, 0, reward_win
