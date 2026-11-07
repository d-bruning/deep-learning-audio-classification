from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    Multiply,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_gatedcnn_model(
    x,
    filters
):

    features = Conv2D(
        filters,
        (3, 3),
        padding="same"
    )(x)

    features = BatchNormalization()(features)

    features = Activation("relu")(features)

    gate = Conv2D(
        filters,
        (3, 3),
        padding="same",
        activation="sigmoid"
    )(x)

    x = Multiply()([
        features,
        gate
    ])

    return x


def build_residualgatedcnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-3
):

    inputs = Input(shape=input_shape)

    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling2D(
        (2, 2)
    )(x)

    #
    # Gated Block 1
    #

    x = gated_block(
        x,
        64
    )

    x = MaxPooling2D(
        (2, 2)
    )(x)

    #
    # Gated Block 2
    #

    x = gated_block(
        x,
        128
    )

    x = MaxPooling2D(
        (2, 2)
    )(x)

    x = GlobalAveragePooling2D()(x)

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
        name="residualgatedcnn"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model