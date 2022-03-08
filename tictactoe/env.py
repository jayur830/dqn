import tensorflow as tf
import numpy as np

from rl.env import Environment
from tictactoe.commons import reward_reset, reward_draw, reward_lose, reward_win, reward_continue, agent, enemy, empty

indices = np.array([[i, j] for i in range(3) for j in range(3)])


class TicTacToeEnvironment(Environment):
    def __init__(self, init_state: np.ndarray):
        super().__init__(init_state)
        self.__iter = 0
        self.__enemy_agent = tf.keras.models.load_model("tictactoe.h5")
        self.__turn = enemy

    def step(self, action: int):
        next_state = self._state.copy()

        self.__iter += 1

        # try:
        #     next_state = self.__step_agent(next_state, action)
        # except:
        #     reward = reward_reset
        #     done = True
        #     self.__iter = 0
        #     return next_state, reward, done, {"status": "RESET"}
        #
        # done, winner, reward, info = self.__result(next_state)
        # if done:
        #     self.__iter = 0
        #     return next_state, reward, done, info
        #
        # next_state = self.__step_random_enemy(next_state, action)
        #
        # done, winner, reward, info = self.__result(next_state)
        # if done:
        #     self.__iter = 0
        #     return next_state, reward, done, info

        try:
            next_state = self.__step_agent(next_state, action)
        except:
            reward = reward_reset
            done = True
            self.__iter = 0
            return next_state, reward, done, { "status": "RESET" }

        done, winner, reward, info = self.__result(next_state)
        if done:
            self.__iter = 0
            return next_state, reward, done, info

        self.__turn = enemy if self.__turn == agent else agent

        return next_state, reward, done, info

    def reset(self):
        self._state = self._init_state.copy()
        index = np.random.randint(9)
        self.__turn = [agent, enemy][np.random.randint(2)]
        # self._state[indices[index][0], indices[index][1]] = self.__turn
        return self._state

    def __step_agent(self, next_state: np.ndarray, action: int):
        if np.any(next_state == empty):
            if next_state[indices[action, 0], indices[action, 1]] != empty:
                raise ValueError()
            else:
                next_state[indices[action, 0], indices[action, 1]] = self.__turn
            self._state = next_state
        return next_state

    def __step_random_enemy(self, next_state: np.ndarray, action: int):
        if np.any(next_state == empty):
            indexes = np.where(next_state.reshape(-1) == 0)[0]
            if indexes.shape[0] > 1:
                indexes = indexes[np.where(indexes != action)[0]]
            index = indexes[np.random.randint(indexes.shape[0])]
            next_state[indices[index][0], indices[index][1]] = enemy
            self._state = next_state
        return next_state

    def __step_human_enemy(self, next_state: np.ndarray):
        if np.any(next_state == empty):
            for i in range(next_state.shape[0]):
                for j in range(next_state.shape[1]):
                    if next_state[i, j, 0] == agent:
                        print("X ", end="")
                    elif next_state[i, j, 0] == enemy:
                        print("O ", end="")
                    else:
                        print(". ", end="")
                print()
            index = int(input()) - 1
            if index == 6:
                index = 0
            elif index == 7:
                index = 1
            elif index == 8:
                index = 2
            elif index == 0:
                index = 6
            elif index == 1:
                index = 7
            elif index == 2:
                index = 8
            next_state[indices[index][0], indices[index][1]] = enemy
            self._state = next_state
        return next_state

    def __step_agent_enemy(self, next_state: np.ndarray, action: int):
        enemy_q_values = np.reshape(self.__enemy_agent(np.reshape(next_state, (1,) + next_state.shape)), -1).copy()
        enemy_q_values[action] = np.min(enemy_q_values)
        indexes = np.where(np.reshape(next_state, -1) != empty)[0]
        enemy_q_values[indexes] = np.min(enemy_q_values)
        enemy_action = np.argmax(enemy_q_values)
        next_state[indices[enemy_action, 0], indices[enemy_action, 1]] = enemy
        return next_state

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
