import tensorflow as tf

from cartpole.common import cartpole_state_dim, cartpole_action_dim


def agent_model(
        kernel_initializer: str = "he_normal",
        learning_rate: float = 0.01):
    input_layer = tf.keras.layers.Input(shape=(cartpole_state_dim))
    x = tf.keras.layers.Dense(
        units=32,
        kernel_initializer=kernel_initializer)(input_layer)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(
        units=32,
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dense(
        units=cartpole_action_dim,
        activation="linear",
        kernel_initializer=kernel_initializer)(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.mean_squared_error)
    model.summary()

    return model
