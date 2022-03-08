import tensorflow as tf
import numpy as np


@tf.function
def randint(size):
    return np.random.randint(size)
