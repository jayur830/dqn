import tensorflow as tf

from grid_world.commons import grid_world_width, grid_world_height


def agent_model(
        kernel_initializer: str = "he_normal",
        learning_rate: float = 0.01):
    input_layer = tf.keras.layers.Input(shape=(grid_world_width, grid_world_height, 1))
    x = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=8,
        kernel_size=3,
        padding="same",
        kernel_initializer=kernel_initializer,
        use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=4,
        kernel_size=1,
        activation="linear",
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.mean_squared_error)
    model.summary()

    return model
