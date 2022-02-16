import os
import tensorflow as tf
import numpy as np

from main import Agent


def agent_model(
        kernel_initializer="he_normal",
        learning_rate=1e-3):
    input_layer = tf.keras.layers.Input(shape=(3, 3, 1))
    x = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
    x = tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(tf.keras.activations.swish)(x)
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        activation="softmax")(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.categorical_crossentropy)
    model.summary()

    return model


if __name__ == "__main__":
    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    #
    # def mask(state: np.ndarray, agent_output: np.ndarray):
    #     state = state.reshape(state.shape[:-1])
    #     q = agent_output.reshape(agent_output.shape[1:-1])
    #     indexes = np.transpose(np.where(state != 0))
    #     for i in range(indexes.shape[0]):
    #         q[indexes[i][0], indexes[i][1]] = 0
    #     return q
    #
    # agent = Agent(model=agent_model())
    # agent.action(state=np.asarray([[[0], [0.5], [0]], [[1], [1], [0]], [[0], [0], [1]]]), mask=mask)

    a = np.asarray([1, 2, 3])
    print(a)
    b = a.copy()
    print(b)
