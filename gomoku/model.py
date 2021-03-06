import tensorflow as tf

from gomoku.commons import gomoku_size


def agent_model(
        kernel_initializer="he_normal",
        learning_rate=0.01):
    input_layer = tf.keras.layers.Input(shape=(gomoku_size, gomoku_size, 1))
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=1,
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        units=gomoku_size * gomoku_size,
        activation="linear",
        kernel_initializer=kernel_initializer)(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.mean_squared_error)
    model.summary()

    return model
