import tensorflow as tf
import numpy as np


@tf.function
def choice(size):
    return np.random.choice(size, size=size, replace=False)


if __name__ == '__main__':
    print(choice(5))
