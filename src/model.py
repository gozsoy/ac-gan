from tensorflow import keras
from tensorflow.keras import layers, Model


class Generator(Model):
    def __init__(self):
        super().__init__()
        self.main = keras.Sequential(
            [
                keras.Input(shape=(128,)),
                layers.Dense(7 * 7 * 128),
                layers.Reshape((7, 7, 128)),
                layers.Conv2DTranspose(128, kernel_size=4,
                                       strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, kernel_size=4,
                                       strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(1, (7, 7), padding="same", activation="tanh"),
            ],
            name="generator",
        )

    def call(self, x):
        return self.main(x)


class Discriminator(Model):
    def __init__(self):
        super().__init__()
        self.main = keras.Sequential(
            [
                keras.Input(shape=(28, 28, 1)),
                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.GlobalMaxPooling2D(),
                layers.Dense(1),
            ],
            name="discriminator",
        )

    def call(self, x):
        return self.main(x)
