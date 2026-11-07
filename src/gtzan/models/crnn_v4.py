from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Reshape,
    LSTM,
    Dropout,
    Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling1D

def build_crnn_model(
    input_shape,
    num_classes,
    learning_rate=1e-4
):

    inputs = Input(shape=input_shape)

    # ------------------------------
    # CNN Feature Extraction
    # ------------------------------

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

    print("Before reshape:", x.shape)

    _, h, w, c = x.shape

    x = Reshape(
        (
            int(w),
            int(h * c)
        )
    )(x)

    print("After reshape:", x.shape)

    x = GlobalAveragePooling1D()(x)
    
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
        inputs=inputs,
        outputs=outputs,
        name="crnn"
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
