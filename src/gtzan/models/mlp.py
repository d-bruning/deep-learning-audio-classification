from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam


def build_mlp_model(
    input_shape,
    num_classes,
    learning_rate=1e-3
):
    """
    MLP baseline.

    Spectrogram image is flattened into a
    feature vector and passed through
    fully connected layers.
    """

    model = Sequential([
    
        Input(
            shape=input_shape
        ),
    
        Flatten(),
    
        Dense(
            64,
            activation="relu"
        ),
    
        Dropout(
            0.5
        ),
    
        Dense(
            32,
            activation="relu"
        ),
    
        Dropout(
            0.5
        ),
    
        Dense(
            16,
            activation="relu"
        ),
    
        Dropout(
            0.3
        ),
    
        Dense(
            num_classes,
            activation="softmax"
        )
    ])
    
    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model