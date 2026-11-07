from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Add,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def residual_block(
    x,
    filters,
    stride=1
):

    shortcut = x

    if (
        x.shape[-1] != filters
        or stride != 1
    ):
        shortcut = Conv2D(
            filters,
            (1, 1),
            strides=stride,
            padding="same"
        )(shortcut)

        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(
        filters,
        (3, 3),
        strides=stride,
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = Conv2D(
        filters,
        (3, 3),
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Add()([
        shortcut,
        x
    ])

    x = Activation("relu")(x)

    return x


def build_shallowresnet_model(
    input_shape,
    num_classes,
    learning_rate=5e-4
):

    inputs = Input(shape=input_shape)

    x = Conv2D(
        32,
        (3, 3),
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = residual_block(
        x,
        32
    )

    x = MaxPooling2D((2, 2))(x)

    x = residual_block(
        x,
        64
    )

    x = MaxPooling2D((2, 2))(x)

    x = residual_block(
        x,
        128
    )

    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        128,
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
        name="shallowresnet"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model