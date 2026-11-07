from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Reshape,
    LSTM,
    Dropout,
    Dense
)
from tensorflow.keras.optimizers import Adam


def build_crnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-4
):
    """
    Convolutional Recurrent Neural Network (CRNN)

    Architecture:
        Conv2D(32)
        BatchNorm
        MaxPool

        Conv2D(64)
        BatchNorm
        MaxPool

        Conv2D(64)
        BatchNorm
        MaxPool

        Reshape

        LSTM(64)

        Dense(128)
        Dropout(0.5)

        Softmax Output

    Purpose:
        Represents the temporal-modeling category
        in the FMA Small duration study.
    """

    inputs = Input(
        shape=input_shape
    )

    # ==========================================
    # CNN Feature Extraction
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

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        (2, 1)
    )(x)

    # ==========================================
    # Convert Feature Map to Sequence
    # ==========================================

    shape = x.shape

    _, height, width, channels = shape

    x = Reshape(
        (
            int(width),
            int(height * channels)
        )
    )(x)

    # ==========================================
    # Temporal Modeling
    # ==========================================

    x = LSTM(
        64,
        return_sequences=False
    )(x)

    # ==========================================
    # Classification Head
    # ==========================================

    x = Dense(
        128,
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
        name="crnn"
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