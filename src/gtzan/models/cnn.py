from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


def build_cnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-3
):
    """
    Original CNN baseline architecture.

    Closely matches the model used in the
    initial experiment while allowing
    configurable input size and class count.
    """

    model = Sequential([

        Input(
            shape=input_shape
        ),

        Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        MaxPooling2D(),

        Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        MaxPooling2D(),

        Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        MaxPooling2D(),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(
            0.5
        ),

        Dense(
            num_classes,
            activation="softmax"
        )

    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
