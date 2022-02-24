import tensorflow as tf


def agent_model(
        kernel_initializer="he_normal",
        learning_rate=0.01):
    input_layer = tf.keras.layers.Input(shape=(3, 3, 1))
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        use_bias=False,
        padding="same",
        kernel_initializer=kernel_initializer)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        use_bias=False,
        padding="same",
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        use_bias=False,
        padding="same",
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.01)(x)
    x = tf.keras.layers.Conv2D(
        filters=1,
        activation="sigmoid",
        kernel_size=1,
        kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Flatten()(x)

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.binary_crossentropy)
    model.summary()

    return model
