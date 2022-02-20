import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.__state_buffer = deque(maxlen=maxlen)
        self.__action_buffer = deque(maxlen=maxlen)
        self.__reward_buffer = deque(maxlen=maxlen)
        self.__next_state_buffer = deque(maxlen=maxlen)

    def put(self, state, action, reward, next_state):
        self.__state_buffer.append(state)
        self.__action_buffer.append(action)
        self.__reward_buffer.append(reward)
        self.__next_state_buffer.append(next_state)

    def sample(self):
        return np.asarray(self.__state_buffer), \
               np.asarray(self.__action_buffer), \
               np.asarray(self.__reward_buffer), \
               np.asarray(self.__next_state_buffer)

    def __getitem__(self, i):
        return self.__state_buffer[i], \
               self.__action_buffer[i], \
               self.__reward_buffer[i], \
               self.__next_state_buffer[i]

    def __len__(self):
        return len(self.__state_buffer)
