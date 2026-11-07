from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_cnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-3
):
    """
    Baseline CNN for FMA Small.

    Purpose:
        Scratch-trained convolutional baseline
        used for duration experiments.

    Architecture:
        Conv(32)
        Conv(64)
        Conv(128)

        GlobalAveragePooling

        Dense(256)
        Dropout(0.5)

        Softmax
    """

    inputs = Input(
        shape=input_shape
    )

    # ==========================================
    # Block 1
    # ==========================================

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    # ==========================================
    # Block 2
    # ==========================================

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    # ==========================================
    # Block 3
    # ==========================================

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    # ==========================================
    # Classification Head
    # ==========================================

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = Dropout(
        0.5
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="cnn"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model