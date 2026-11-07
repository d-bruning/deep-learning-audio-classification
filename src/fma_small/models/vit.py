from tensorflow.keras import Model

from tensorflow.keras.layers import (
    Input,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Dropout,
    GlobalAveragePooling1D,
    Embedding,
    Reshape
)

from tensorflow.keras.optimizers import Adam

import tensorflow as tf


# =====================================================
# PATCH ENCODER
# =====================================================

class PatchEmbedding(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        patch_size,
        embed_dim
    ):

        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = Dense(
            embed_dim
        )

    def call(
        self,
        images
    ):

        patches = tf.image.extract_patches(
            images=images,
            sizes=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            strides=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            rates=[
                1,
                1,
                1,
                1
            ],

            padding="VALID"
        )

        batch_size = tf.shape(
            patches
        )[0]

        patches = tf.reshape(
            patches,
            [
                batch_size,
                -1,
                patches.shape[-1]
            ]
        )

        return self.projection(
            patches
        )


# =====================================================
# VISION TRANSFORMER
# =====================================================

def build_vit_model(
    input_shape,
    num_classes,
    learning_rate=1e-4,
    patch_size=16,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    mlp_dim=256,
    dropout_rate=0.1
):

    inputs = Input(
        shape=input_shape
    )

    # ---------------------------------------------
    # Patch Embedding
    # ---------------------------------------------

    patches = PatchEmbedding(
        patch_size=patch_size,
        embed_dim=embed_dim
    )(inputs)

    num_patches = (
        (input_shape[0] // patch_size)
        *
        (input_shape[1] // patch_size)
    )

    positions = tf.range(
        start=0,
        limit=num_patches,
        delta=1
    )

    position_embedding = Embedding(
        input_dim=num_patches,
        output_dim=embed_dim
    )(positions)

    x = patches + position_embedding

    # ---------------------------------------------
    # Transformer Encoder Blocks
    # ---------------------------------------------

    for _ in range(
        num_layers
    ):

        attention_input = (
            LayerNormalization(
                epsilon=1e-6
            )(x)
        )

        attention_output = (
            MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embed_dim
            )(
                attention_input,
                attention_input
            )
        )

        x = Add()([
            x,
            attention_output
        ])

        mlp_input = (
            LayerNormalization(
                epsilon=1e-6
            )(x)
        )

        mlp_output = Dense(
            mlp_dim,
            activation="gelu"
        )(mlp_input)

        mlp_output = Dropout(
            dropout_rate
        )(mlp_output)

        mlp_output = Dense(
            embed_dim
        )(mlp_output)

        x = Add()([
            x,
            mlp_output
        ])

    # ---------------------------------------------
    # Classification Head
    # ---------------------------------------------

    x = LayerNormalization(
        epsilon=1e-6
    )(x)

    x = GlobalAveragePooling1D()(
        x
    )

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
        name="vit"
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