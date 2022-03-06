import numpy as np

from rl.env import Environment
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win, reward_continue, agent, enemy, empty

indices = [[i, j] for i in range(3) for j in range(3)]


class TicTacToeEnvironment(Environment):
    def __init__(self, init_state: np.ndarray):
        super().__init__(init_state)
        self.__iter = 0

    def step(self, action: int):
        next_state = self._state.copy()

        self.__iter += 1
        if not next_state.all():
            if next_state[indices[action][0], indices[action][1]] != empty:
                reward = reward_reset
                done = True
                self.__iter = 0
                return next_state, reward, done, { "status": "RESET" }
            else:
                next_state[indices[action][0], indices[action][1]] = agent
            self._state = next_state

        done, winner, reward, info = self.__result(next_state)
        if done:
            self.__iter = 0
            return next_state, reward, done, info

        if not next_state.all():
            indexes = np.where(next_state.reshape(-1) == 0)[0]
            if indexes.shape[0] > 1:
                indexes = indexes[np.where(indexes != action)[0]]
            index = indexes[np.random.randint(indexes.shape[0])]
            next_state[indices[index][0], indices[index][1]] = enemy
            self._state = next_state

        done, winner, reward, info = self.__result(next_state)
        if done:
            self.__iter = 0
            return next_state, reward, done, info

        return next_state, reward, done, info

    def reset(self):
        self._state = self._init_state.copy()
        index = np.random.randint(9)
        self._state[indices[index][0], indices[index][1]] = enemy
        return self._state

    def __result(self, next_state: np.ndarray):
        info = {
            "status": ""
        }
        vertical_sum_state, horizontal_sum_state = next_state.sum(axis=0), next_state.sum(axis=1)
        if enemy * 3 in vertical_sum_state or enemy * 3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == enemy * 3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == enemy * 3:
            info["status"] = "LOSE"
            return True, enemy, reward_lose, info
        elif agent * 3 in vertical_sum_state or agent * 3 in horizontal_sum_state \
                or next_state[0, 0] + next_state[1, 1] + next_state[2, 2] == agent * 3 \
                or next_state[0, 2] + next_state[1, 1] + next_state[2, 0] == agent * 3:
            info["status"] = "WIN"
            return True, agent, (4 - self.__iter) * .1 + reward_win, info
        elif np.where(next_state.reshape(-1) == empty)[0].shape[0] == 0:
            info["status"] = "DRAW"
            return True, 0, reward_draw, info
        else:
            info["status"] = "PLAY"
            return False, 0, reward_continue, info
