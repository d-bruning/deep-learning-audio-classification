from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Reshape,
    GRU,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_grucnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-4
):

    inputs = Input(
        shape=input_shape
    )

    #
    # CNN Feature Extraction
    #

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

    #
    # Convert to sequence
    #

    _, h, w, c = x.shape

    x = Reshape(
        (
            int(w),
            int(h * c)
        )
    )(x)

    #
    # GRU
    #

    x = GRU(
        32,
        dropout=0.3,
        recurrent_dropout=0.3
    )(x)

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
        inputs,
        outputs,
        name="grucnn"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model