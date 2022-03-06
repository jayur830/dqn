import tensorflow as tf
import numpy as np


@tf.function
def randint(size):
    return np.random.randint(size)


@tf.function
def random_indexes(size: int):
    return np.random.choice(size, size=size, replace=False)
