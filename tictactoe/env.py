import random
import numpy as np

from rl.env import Environment
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win, reward_continue

indices = [[i, j] for i in range(3) for j in range(3)]


class TicTacToeEnvironment(Environment):
    def step(self, action: int):
        next_state = self._state.copy()

        if not next_state.all():
            if next_state[indices[action][0], indices[action][1]] != 0:
                reward = reward_reset
                done = True
                return next_state, reward, done, { "status": "RESET" }
            else:
                next_state[indices[action][0], indices[action][1]] = 1.
            self._state = next_state

        done, winner, reward, info = self.__result(next_state)
        if done:
            return next_state, reward, done, info

        if not next_state.all():
            indexes = np.where(next_state.reshape(-1) == 0)[0]
            if indexes.shape[0] > 1:
                indexes = indexes[np.where(indexes != action)[0]]
            index = indexes[np.random.randint(indexes.shape[0])]
            next_state[indices[index][0], indices[index][1]] = -1.
            self._state = next_state

        done, winner, reward, info = self.__result(next_state)
        if done:
            return next_state, reward, done, info

        return next_state, reward, done, info

    def reset(self):
        self._state = self._init_state.copy()
        index = np.random.randint(9)
        self._state[indices[index][0], indices[index][1]] = -1.
        return self._state

    def __result(self, next_state: np.ndarray):
        info = {
            "status": ""
        }
        vertical_sum_state, horizontal_sum_state = next_state.sum(axis=0), next_state.sum(axis=1)
        if -3 in vertical_sum_state or -3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == -3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == -3:
            info["status"] = "LOSE"
            return True, -1, reward_lose, info
        elif 3 in vertical_sum_state or 3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == 3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == 3:
            info["status"] = "WIN"
            return True, 1, reward_win, info
        elif np.where(next_state.reshape(-1) == 0)[0].shape[0] == 0:
            info["status"] = "DRAW"
            return True, 0, reward_draw, info
        else:
            info["status"] = "PLAY"
            return False, 0, reward_continue, info
