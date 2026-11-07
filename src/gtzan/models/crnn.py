from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Reshape,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_crnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-4
):
    """
    CRNN for music genre classification.

    Spectrogram width is treated as the
    temporal dimension presented to the LSTM.
    """

    inputs = Input(
        shape=input_shape
    )

    # ---------------------------------
    # CNN Feature Extraction
    # ---------------------------------

    x = Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    x = Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    #
    # IMPORTANT:
    # no third pooling layer
    #
    # preserve more time resolution
    #

    # ---------------------------------
    # CNN -> Sequence conversion
    # ---------------------------------

    _, h, w, c = x.shape

    x = Reshape(
        (
            int(w),
            int(h * c)
        )
    )(x)

    # ---------------------------------
    # Recurrent Layer
    # ---------------------------------

    x = Bidirectional(
        LSTM(
            128,
            return_sequences=False,
            recurrent_dropout=0.2
        )
    )(x)

    # ---------------------------------
    # Classification Head
    # ---------------------------------

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = Dropout(
        0.5
    )(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(
        0.3
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
        metrics=["accuracy"]
    )

    return model