import numpy as np

from rl.env import Environment
from gomoku.commons import gomoku_width, gomoku_height, reward_reset, reward_draw, reward_lose, reward_win, reward_continue, black, white, empty

indices = [[i, j] for i in range(gomoku_height) for j in range(gomoku_width)]


class TicTacToeEnvironment(Environment):
    def __init__(self, init_state: np.ndarray):
        super().__init__(init_state)
        self.__iter = 0
        self.__first_is = 0
        self.__agent_is = 0

    def step(self, action: int):
        next_state = self._state.copy()
        self.__iter += 1

        next_state = self.__step_agent(next_state, action) if self.__first_is != self.__agent_is else self.__step_random_enemy(next_state, action)

        done, winner, reward, info = self.__result(next_state)
        if done:
            self.__iter = 0
            return next_state, reward, done, info

        next_state = self.__step_random_enemy(next_state, action) if self.__first_is != self.__agent_is else self.__step_agent(next_state, action)

        done, winner, reward, info = self.__result(next_state)
        if done:
            self.__iter = 0
            return next_state, reward, done, info

        return next_state, reward, done, info

    def reset(self):
        self._state = self._init_state.copy()
        self.__first_is = [black, white][np.random.randint(2)]
        self._state[self._state.shape[0] // 2, self._state.shape[1] // 2] = self.__first_is
        self.__agent_is = [black, white][np.random.randint(2)]
        return self._state

    def __step_agent(self, next_state: np.ndarray, action: int):
        if np.sum(next_state == empty) > 0:
            if next_state[indices[action][0], indices[action][1]] != empty:
                reward = reward_reset
                done = True
                self.__iter = 0
                return next_state, reward, done, { "status": "RESET" }
            else:
                next_state[indices[action][0], indices[action][1]] = self.__agent_is
            self._state = next_state
        return next_state

    def __step_random_enemy(self, next_state: np.ndarray, action: int):
        if np.sum(next_state == empty) > 0:
            indexes = np.where(next_state.reshape(-1) == empty)[0]
            if indexes.shape[0] > 1:
                indexes = indexes[np.where(indexes != action)[0]]
            index = indexes[np.random.randint(indexes.shape[0])]
            next_state[indices[index][0], indices[index][1]] = white if self.__agent_is == black else black
            self._state = next_state
        return next_state

    def __step_minimax_enemy(self, next_state: np.ndarray, action: int):
        pass

    def __result(self, next_state: np.ndarray):
        info = {
            "status": ""
        }
        next_state = next_state.reshape(next_state.shape[:-1])
        for i in range(next_state.shape[0] - 4):
            if np.sum(np.sum(next_state[i:i + 5], axis=0) == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win if self.__agent_is == black else reward_lose, info
            elif np.sum(np.sum(next_state[i:i + 5], axis=0) == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win if self.__agent_is == white else reward_lose, info
        for j in range(next_state.shape[1] - 4):
            if np.sum(np.sum(next_state[:, j:j + 5], axis=1) == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win if self.__agent_is == black else reward_lose, info
            elif np.sum(np.sum(next_state[:, j:j + 5], axis=1) == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win if self.__agent_is == white else reward_lose, info
        state_affine1 = self.__affine1(next_state)
        for i in range(state_affine1.shape[0] - 4):
            if np.sum(np.sum(state_affine1[i:i + 5], axis=0) == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win if self.__agent_is == black else reward_lose, info
            elif np.sum(np.sum(state_affine1[i:i + 5], axis=0) == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win if self.__agent_is == white else reward_lose, info
        for j in range(state_affine1.shape[1] - 4):
            if np.sum(np.sum(state_affine1[:, j:j + 5], axis=1) == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win if self.__agent_is == black else reward_lose, info
            elif np.sum(np.sum(state_affine1[:, j:j + 5], axis=1) == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win if self.__agent_is == white else reward_lose, info
        state_affine2 = self.__affine2(next_state)
        for i in range(state_affine2.shape[0] - 4):
            if np.sum(np.sum(state_affine2[i:i + 5], axis=0) == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win if self.__agent_is == black else reward_lose, info
            elif np.sum(np.sum(state_affine2[i:i + 5], axis=0) == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win if self.__agent_is == white else reward_lose, info
        for j in range(state_affine2.shape[1] - 4):
            if np.sum(np.sum(state_affine2[:, j:j + 5], axis=1) == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win if self.__agent_is == black else reward_lose, info
            elif np.sum(np.sum(state_affine2[:, j:j + 5], axis=1) == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win if self.__agent_is == white else reward_lose, info
        info["status"] = "PLAY"
        return False, 0, reward_continue, info

    def __affine1(self, x):
        w, h = x.shape[1], x.shape[0]
        _x = np.zeros(shape=(2 * h - 1, w))
        for col in range(w):
            _x[col:col + h, col] = x[:, col]
        x = _x
        return x

    def __affine2(self, x):
        w, h = x.shape[1], x.shape[0]
        _x = np.zeros(shape=(2 * h - 1, w))
        for col in range(w):
            _x[-h - col:2 * h - 1 - col, col] = x[:, col]
        x = _x
        return x
