from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
import tensorflow_hub as hub


def build_ast_model(
    input_shape,
    num_classes,
    learning_rate=1e-5
):
    """
    Audio Spectrogram Transformer (AST)

    Purpose:
        Transformer-based architecture for the
        FMA Small duration study.

    Architecture:
        Pretrained Vision Transformer Backbone
        GlobalAveragePooling1D
        Dense(256)
        Dropout(0.5)
        Softmax Output

    Category:
        Transformer

    Training:
        Pretrained
        Fine-tuning with low learning rate.
    """

    inputs = Input(
        shape=input_shape
    )

    # ------------------------------------------
    # Vision Transformer Backbone
    # ------------------------------------------

    vit_layer = hub.KerasLayer(
        "https://tfhub.dev/sayakpaul/vit_b16_fe/1",
        trainable=False
    )

    x = vit_layer(inputs)

    x = GlobalAveragePooling1D()(x)

    # ------------------------------------------
    # Classification Head
    # ------------------------------------------

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
        name="ast"
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