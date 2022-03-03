import numpy as np

from rl.env import Environment
from grid_world.commons import \
    reward_lose, \
    reward_win, \
    reward_continue, \
    grid_world_width, \
    grid_world_height


class GridWorldEnvironment(Environment):
    def step(self, action: int):
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
            self._reward = reward_lose
            info["status"] = "LOSE"
        elif current_index[0] == goal_index[0] and current_index[1] == goal_index[1]:
            done = True
            self._reward = reward_win
            info["status"] = "WIN"
            next_state[current_index[0], current_index[1]] = 1.
        else:
            self._reward = reward_continue
            next_state[current_index[0], current_index[1]] = 1.
            done = False

        self._state = next_state
        return next_state, self._reward, done, info

    def reset(self):
        self._state = self._init_state.copy()
        self._state[0, 0, 0] = 1.
        self._state[grid_world_height - 1, grid_world_width - 1, 0] = 2.
        return self._state
