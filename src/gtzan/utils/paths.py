from pathlib import Path


# ==================================================
# Project Root
# ==================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)


# ==================================================
# Data Directories
# ==================================================

DATA_DIR = (
    PROJECT_ROOT /
    "data" /
    "gtzan"
)

AUDIO_DIR = (
    DATA_DIR /
    "audio_files"
)

METADATA_DIR = (
    DATA_DIR /
    "metadata"
)

SPECTROGRAM_DIR = (
    DATA_DIR /
    "spectrograms"
)


# ==================================================
# GTZAN Dataset Paths
# ==================================================

GTZAN_AUDIO_DIR = (
    AUDIO_DIR /
    "source"
)

GTZAN_SPECTROGRAM_DIR = (
    SPECTROGRAM_DIR
)

GTZAN_SPLIT_FILE = (
    METADATA_DIR /
    "song_splits.csv"
)


# ==================================================
# Source Code Locations
# ==================================================

MODELS_DIR = (
    PROJECT_ROOT /
    "src" /
    "gtzan" /
    "models"
)

PROCESSING_DIR = (
    PROJECT_ROOT /
    "src" /
    "gtzan" /
    "processing"
)


# ==================================================
# Results
# ==================================================

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "gtzan"
)


# ==================================================
# Analytics
# ==================================================

ANALYTICS_DIR = (
    PROJECT_ROOT /
    "analytics" /
    "gtzan"
)


# ==================================================
# WandB
# ==================================================

WANDB_DIR = (
    PROJECT_ROOT /
    "wandb"
)


# ==================================================
# Output Path Factory
# ==================================================

def get_output_paths(
    model_name
):
    """
    Creates and returns:

    results/gtzan/<model_name>/*

    Returns:
        {
            "root",
            "checkpoints",
            "confusion_matrices",
            "duration_results",
            "genre_results",
            "histories",
            "models",
            "metadata"
        }
    """

    