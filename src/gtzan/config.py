# modify MODEL_NAME to select the model for training
MODEL_NAME = "mlp"
#    "mlp"
#    "cnn"
#    "mobilenetv2"
#    "vgg"
#    "shallowresnet"
#    "residualcnn"
#    "crnn"
#    "capsnet"


SEEDS_TEST = [42]
SEEDS = [
    42,
    123,
    999,
    314,
    2024
]

DURATIONS_TEST = ["15s"]
DURATIONS = [
    "1s",
    "3s",
    "5s",
    "10s",
    "15s",
    "20s",
    "25s",
    "30s"
]

CLASS_NAMES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

NUM_CLASSES = len(CLASS_NAMES)

GENRE_TO_INT = {
    genre: idx
    for idx, genre in enumerate(CLASS_NAMES)
}

IMG_SIZE = (128, 128)
#IMG_SIZE = (224, 224)

BATCH_SIZE = 32

EPOCHS = 30

LEARNING_RATES = {
    "cnn": 1e-3,
    "mobilenetv2": 1e-4,
    "vgg": 1e-3,
    "shallowresnet": 5e-4,
    "residualcnn": 1e-4,
    "crnn": 1e-4,
    "capsnet": 1e-3,
    "mlp": 1e-4
}