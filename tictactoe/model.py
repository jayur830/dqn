import tensorflow as tf


def agent_model(
        kernel_initializer="he_uniform",
        learning_rate=0.001):
    input_layer = tf.keras.layers.Input(shape=(3, 3, 1))
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=kernel_initializer)(input_layer)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.ELU()(x)
    x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, kernel_initializer=kernel_initializer)(x)
    x = tf.keras.layers.Flatten()(x)
    v = tf.keras.layers.Dense(units=1, activation="linear", kernel_initializer=kernel_initializer)(x)
    a = tf.keras.layers.Dense(units=9, activation="linear", kernel_initializer=kernel_initializer)(x)
    a_mean = tf.reduce_mean(a)
    x = v + a - a_mean

    model = tf.keras.models.Model(input_layer, x)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.mean_squared_error)
    model.summary()

    return model
