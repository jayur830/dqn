import random
import numpy as np

from rl.env import Environment
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win

indices = [[i, j] for i in range(3) for j in range(3)]


class TicTacToeEnvironment(Environment):
    def step(self, action: int):
        next_state = self._state.copy()
        info = {
            "status": ""
        }

        self._done, winner, self._reward = self.__result(next_state)
        if self._done:
            if winner == -1:
                info["status"] = "LOSE"
            elif winner == 1:
                info["status"] = "WIN"
            else:
                info["status"] = "DRAW"
            return self._reward, next_state, info

        if not next_state.all():
            index = int(action)
            self._reward, next_state, info = self.__play(next_state, index)

        self._state = next_state
        return next_state, self._reward, self._done, info

    def _state_preprocess(self, state: np.ndarray):
        indexes = np.transpose(np.where(state == 0))
        if indexes.shape[0] > 0:
            random_index = np.random.randint(indexes.shape[0])
            state[indexes[random_index, 0], indexes[random_index, 1]] = -1.
        return state

    def __play(self, next_state, index):
        info = {
            "status": ""
        }
        if next_state[indices[index][0], indices[index][1]] != 0:
            print(next_state)
            input()
            self._reward = reward_reset
            self._done = True
            info["status"] = "RESET"
            return self._reward, next_state, info
        next_state[indices[index][0], indices[index][1]] = 1.
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
        elif np.where(next_state.reshape(-1) == 0)[0].shape[0] == 0:
            return True, 0, reward_draw
        else:
            return False, 0, reward_win
