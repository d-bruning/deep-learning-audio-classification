"""
FMA Small Configuration

Study:
    V2 Music Classification Clip Length and Genre Study

Research Goal:
    Determine whether the diminishing-return
    relationship between audio duration and
    genre classification performance observed
    in GTZAN persists on the larger FMA Small
    dataset and across distinct model categories.
"""


# ==================================================
# Experiment Selection
# ==================================================

RUN_MODE = "full"

MODEL_NAME = "vit"

# Available Models
#
# "cnn"
# "mobilenetv2"
# "crnn"
# "ast"
# "vit"

# ==================================================
# Dataset
# ==================================================

DATASET_NAME = "fma_small"

CLASS_NAMES = [
    "electronic",
    "experimental",
    "folk",
    "hip-hop",
    "instrumental",
    "international",
    "pop",
    "rock"
]

NUM_CLASSES = len(
    CLASS_NAMES
)

GENRE_TO_INT = {
    genre: idx
    for idx, genre
    in enumerate(CLASS_NAMES)
}

INT_TO_GENRE = {
    idx: genre
    for genre, idx
    in GENRE_TO_INT.items()
}


# ==================================================
# Random Seeds
# ==================================================

SEEDS_TEST = [
    42
]

SEEDS = [
    42,
    123,
    314,
    999,
    2024
]


# ==================================================
# Audio Durations
# ==================================================

DURATIONS_TEST = [
    "15s"
]

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


# ==================================================
# Image Parameters
# ==================================================

IMG_SIZE = {
    "cnn": (128,128),
    "mobilenetv2": (224,224),
    "crnn": (128,128),
    "ast": (224, 224),
    "vit": (224, 224)
}


# ==================================================
# Training Parameters
# ==================================================

BATCH_SIZE = 32

EPOCHS = 30

PATIENCE = 5


# ==================================================
# Learning Rates
# ==================================================

LEARNING_RATES = {

    # Scratch CNN
    "cnn": 1e-3,

    # Transfer Learning
    "mobilenetv2": 1e-4,

    # CNN + BiLSTM
    "crnn": 1e-4,

    # Audio Transformer
    "ast": 1e-5,   

    # Vision Transformer
    "vit": 1e-4
}


# ==================================================
# Model Categories
# ==================================================

MODEL_CATEGORIES = {

    "cnn":
        "scratch_cnn",

    "mobilenetv2":
        "pretrained_cnn",

    "crnn":
        "temporal_model",
    
    "ast":
        "audio_transformer",

    "vit":
        "vision_transformer"
}


# ==================================================
# Transfer Learning
# ==================================================

PRETRAINED_MODELS = {
    "cnn": False,
    "mobilenetv2": True,
    "crnn": False,
    "ast": True,
    "vit": True
}


# ==================================================
# WandB
# ==================================================

WANDB_ENABLED = True

WANDB_PROJECT = (
    "audio-classification-comparison"
)

WANDB_ENTITY = None


# ==================================================
# Study Metadata
# ==================================================

STUDY_NAME = (
    "fma_small_v2"
)

STUDY_DESCRIPTION = (
    "Extension study evaluating whether "
    "audio-duration diminishing returns "
    "observed on GTZAN replicate on "
    "FMA Small."
)