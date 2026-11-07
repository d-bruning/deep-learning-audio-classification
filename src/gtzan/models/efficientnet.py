from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input
)


def build_efficientnet_model(
    input_shape,
    num_classes,
    learning_rate=1e-4
):
    """
    EfficientNetB0 transfer learning model for
    music genre classification.
    """

    inputs = Input(
        shape=input_shape
    )

    # EfficientNet preprocessing
    x = preprocess_input(inputs)

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # First pass: frozen backbone
    base_model.trainable = True

    x = base_model(
        x,
        training=False
    )

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
        outputs=outputs
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
