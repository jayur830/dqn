import tensorflow as tf
import numpy as np

from rl.env import Environment
from grid_world.commons import \
    reward_lose, \
    reward_win, \
    reward_continue, \
    grid_world_width, \
    grid_world_height


class GridWorldEnvironment(Environment):
    def __init__(self, init_state: np.ndarray):
        super().__init__(init_state)
        self.__iter = 0
        self.__max_iter = grid_world_width + grid_world_height + 9

    def step(self, action: int):
        self.__iter += 1
        current_index = np.transpose(np.where(self._state == 1.))[0]
        goal_index = np.transpose(np.where(self._state == 2.))[0]
        info = {
            "status": ""
        }

        next_state = self._state.copy()
        next_state[current_index[0], current_index[1]] = 0

        if action == 0:
            current_index[0] -= 1
        elif action == 1:
            current_index[0] += 1
        elif action == 2:
            current_index[1] -= 1
        elif action == 3:
            current_index[1] += 1

        if current_index[0] < 0 \
                or current_index[0] >= self._state.shape[0] \
                or current_index[1] < 0 \
                or current_index[1] >= self._state.shape[1]:
            done = True
            reward = reward_lose
            info["status"] = "LOSE"
            self.__iter = 0
        elif current_index[0] == goal_index[0] and current_index[1] == goal_index[1]:
            done = True
            reward = self.__max_iter - self.__iter
            info["status"] = "WIN"
            next_state[current_index[0], current_index[1]] = 1.
            self.__iter = 0
        else:
            obstacle_points = np.transpose(np.where(next_state == -1.))[:, :-1]
            current_is_obstacle = False
            for i in range(obstacle_points.shape[0]):
                if current_index[0] == obstacle_points[i, 0] and current_index[1] == obstacle_points[i, 1]:
                    current_is_obstacle = True
                    break

            if current_is_obstacle or self.__iter == self.__max_iter:
                done = True
                reward = reward_lose
                info["status"] = "LOSE"
                self.__iter = 0
            else:
                done = False
                reward = reward_continue
                next_state[current_index[0], current_index[1]] = 1.

        self._state = next_state
        return next_state, reward, done, info

    def reset(self):
        self._state = self._init_state.copy()
        # row_indexes = np.random.choice(grid_world_height, size=2, replace=False)
        # col_indexes = np.random.choice(grid_world_width, size=2, replace=False)
        # self._state[row_indexes[0], col_indexes[0], 0] = 1.
        # self._state[row_indexes[1], col_indexes[1], 0] = 2.
        self._state[0, 0, 0] = 1.
        self._state[2, 1, 0] = -1.
        self._state[0, 3, 0] = -1.
        self._state[1, 4, 0] = -1.
        self._state[3, 2, 0] = -1.
        self._state[4, 0, 0] = -1.
        self._state[5, 3, 0] = -1.
        self._state[6, 1, 0] = -1.
        self._state[3, 6, 0] = -1.
        self._state[grid_world_height - 1, grid_world_width - 1, 0] = 2.
        return self._state
