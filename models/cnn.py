from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D,
)


class CNN:
    def __init__(
        self,
        input_shape: tuple,
        num_of_class: int = 1,
        method: str = "softmax",
    ):
        input_tensor = Input(input_shape)

        x = input_tensor

        layers = [
            Conv2D(16, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation="relu"),
            Dropout(0.3),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(num_of_class, activation=method),
        ]

        for layer in layers:
            x = layer(x)

        self.model = Model(inputs=input_tensor, outputs=x)
