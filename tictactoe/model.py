import tensorflow as tf


def agent_model(
        kernel_initializer="he_uniform",
        learning_rate=0.005):
    input_layer = tf.keras.layers.Input(shape=(3, 3, 1))
    x = tf.keras.layers.Flatten()(input_layer)
    x = tf.keras.layers.Dense(
        units=64,
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Dense(
        units=64,
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Dense(
        units=9,
        activation="linear",
        kernel_initializer=kernel_initializer)(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.huber)
    model.summary()

    return model
