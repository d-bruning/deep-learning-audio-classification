from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


def build_mobilenetv2_model(
    input_shape,
    num_classes,
    learning_rate=1e-4
):
    """
    MobileNetV2 music genre classifier.

    Transfer learning architecture using an
    ImageNet-pretrained MobileNetV2 backbone.

    Architecture:
        MobileNetV2
        GlobalAveragePooling2D
        Dense(256)
        Dropout(0.5)
        Dense(num_classes)
    """

    inputs = Input(
        shape=input_shape
    )

    backbone = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # Freeze backbone initially
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
        name="mobilenetv2"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model