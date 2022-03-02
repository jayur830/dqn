import random
import numpy as np

from collections import deque


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.__buffer_size = maxlen
        self.__buffer = deque(maxlen=maxlen)
        self.__size = 0

    def put(self, state, action, reward, next_state, done):
        self.__buffer.append((state, action, reward, next_state, done))
        self.__size += 1

    def sample(self, sample_size: int = 500):
        samples = random.sample(self.__buffer, sample_size)
        return map(lambda x: np.vstack(x).astype(np.float), np.transpose(samples))

    def __getitem__(self, i):
        return self.__buffer[i]

    def __len__(self):
        return self.__size
