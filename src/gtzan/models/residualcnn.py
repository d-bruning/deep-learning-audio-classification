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
from tensorflow.keras.regularizers import l2

def residual_block(
    x,
    filters
):
    """
    Basic residual block.

    Input:
        x

    Output:
        x + F(x)
    """

    shortcut = x

    x = Conv2D(
        filters,
        (3, 3),
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


def build_residualcnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-3
):
    """
    Lightweight Residual CNN
    for music genre classification.

    Designed as a middle ground
    between the baseline CNN and
    large pretrained ResNet models.
    """

    inputs = Input(
        shape=input_shape
    )

    #
    # Stem
    #

    x = Conv2D(
        32,
        (3, 3),
        padding="same"
    )(inputs)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    #
    # Residual Block Group 1
    #

    x = Conv2D(
        64,
        (3, 3),
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = residual_block(
        x,
        64
    )

    x = MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    #
    # Residual Block Group 2
    #

    x = Conv2D(
        128,
        (3, 3),
        padding="same"
    )(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x)

    x = residual_block(
        x,
        128
    )

    x = MaxPooling2D(
        pool_size=(2, 2)
    )(x)

    #
    # Classification Head
    #

    x = GlobalAveragePooling2D()(x)

    x = Dense(
        128,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = Dropout(
        0.6
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="residualcnn"
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