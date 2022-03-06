import tensorflow as tf
import numpy as np

from rl.env import Environment
from gomoku.commons import gomoku_size, reward_reset, reward_draw, reward_lose, reward_win, reward_continue, black, white, empty

indices = np.array([[i, j] for i in range(gomoku_size) for j in range(gomoku_size)])


class GomokuEnvironment(Environment):
    def __init__(self, init_state: np.ndarray):
        super().__init__(init_state)
        self.__iter = 0
        self.__max_iter = gomoku_size * 4 - (gomoku_size % 2 != 0)
        self.__first_is = 0
        self.__agent_is = 0

    def step(self, action: int):
        next_state = self._state.copy()
        self.__iter += 1

        try:
            next_state = self.__step_agent(next_state, action) if self.__first_is != self.__agent_is else self.__step_random_enemy(next_state, action)
        except:
            reward = reward_reset
            done = True
            self.__iter = 0
            return next_state, reward, done, { "status": "RESET" }

        done, winner, reward, info = self.__result(next_state)
        if done:
            self.__iter = 0
            return next_state, reward, done, info

        try:
            next_state = self.__step_random_enemy(next_state, action) if self.__first_is != self.__agent_is else self.__step_agent(next_state, action)
        except:
            reward = reward_reset
            done = True
            self.__iter = 0
            return next_state, reward, done, { "status": "RESET" }

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
        if np.any(next_state == empty):
            if next_state[indices[action, 0], indices[action, 1]] != empty:
                raise ValueError()
            else:
                next_state[indices[action, 0], indices[action, 1]] = self.__agent_is
            self._state = next_state
        return next_state

    def __step_random_enemy(self, next_state: np.ndarray, action: int):
        if np.any(next_state == empty):
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
        next_state = np.reshape(next_state, np.shape(next_state)[:-1])
        for i in range(np.shape(next_state)[0] - 4):
            summary = np.sum(next_state[i:i + 5], axis=0)
            if np.sum(summary == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == black else reward_lose, info
            elif np.sum(summary == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == white else reward_lose, info
        for j in range(next_state.shape[1] - 4):
            summary = np.sum(next_state[:, j:j + 5], axis=1)
            if np.sum(summary == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == black else reward_lose, info
            elif np.sum(summary == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == white else reward_lose, info
        state_affine1, state_affine2 = self.__affine(next_state)
        for i in range(state_affine1.shape[0] - 4):
            summary = np.sum(state_affine1[i:i + 5], axis=0)
            if np.sum(summary == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == black else reward_lose, info
            elif np.sum(summary == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == white else reward_lose, info
        for j in range(state_affine1.shape[1] - 4):
            summary = np.sum(state_affine1[:, j:j + 5], axis=1)
            if np.sum(summary == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == black else reward_lose, info
            elif np.sum(summary == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == white else reward_lose, info
        for i in range(state_affine2.shape[0] - 4):
            summary = np.sum(state_affine2[i:i + 5], axis=0)
            if np.sum(summary == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == black else reward_lose, info
            elif np.sum(summary == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == white else reward_lose, info
        for j in range(state_affine2.shape[1] - 4):
            summary = np.sum(state_affine2[:, j:j + 5], axis=1)
            if np.sum(summary == black * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == black else "LOSE"
                return True, black, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == black else reward_lose, info
            elif np.sum(summary == white * 5) > 0:
                info["status"] = "WIN" if self.__agent_is == white else "LOSE"
                return True, white, reward_win + (self.__max_iter - self.__iter) if self.__agent_is == white else reward_lose, info
        if np.any(next_state == empty):
            info["status"] = "PLAY"
            return False, 0, reward_continue, info
        else:
            info["status"] = "DRAW"
            return True, 0, reward_draw, info

    def __affine(self, x):
        w, h = x.shape[1], x.shape[0]
        a = np.zeros(shape=(2 * h - 1, w))
        b = np.zeros(shape=(2 * h - 1, w))
        for col in range(w):
            a[col:col + h, col] = x[:, col]
            b[-h - col:2 * h - 1 - col, col] = x[:, col]
        return a, b
