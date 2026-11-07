from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Reshape,
    Dense,
    GlobalAveragePooling2D,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_capsnet_model(
    input_shape,
    num_classes,
    learning_rate=1e-3
):

    inputs = Input(shape=input_shape)

    x = Conv2D(
        64,
        (5, 5),
        activation="relu",
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = Conv2D(
        128,
        (5, 5),
        activation="relu",
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        256,
        activation="relu"
    )(x)

    x = Dropout(0.5)(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="capsnet"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model