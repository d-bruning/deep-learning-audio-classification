from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50


def build_resnet50_model(
    input_shape,
    num_classes,
    learning_rate=1e-3
):

    inputs = Input(
        shape=input_shape
    )

    backbone = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    backbone.trainable = False

    x = GlobalAveragePooling2D()(
        backbone.output
    )

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
        name="resnet50"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model