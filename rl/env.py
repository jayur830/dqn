import numpy as np
import random

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


class Environment:
    def __init__(self, init_state: np.ndarray):
        self.__init_state = init_state
        self.__reward = 0
        self.__state = self.__init_state.copy()
        self.__done = True

    def state(self):
        return self.__state.copy()

    def step(self, action):
        next_state = self.__state.copy()
        while True:
            random_index = random.randint(0, 9)
            if not self.__exist(random_index):
                break
        if self.__exist(action):
            self.__reward = -100
        else:
            next_state[indices[action][0], indices[action][1]] = 1.
            next_state[indices[random_index][0], indices[random_index][1]] = -1.
            self.__state = next_state
            self.__done, winner = self.__result(next_state)
            if winner == -1:
                self.__reward = -10
            else:
                self.__reward = max(self.__reward, 0) + 1
        return self.__reward, next_state

    def done(self):
        return self.__done

    def reset(self):
        self.__state = self.__init_state.copy()
        self.__done = False

    def __exist(self, action):
        return self.__state[indices[action][0], indices[action][1]] != 0

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
        else:
            return False, 0
