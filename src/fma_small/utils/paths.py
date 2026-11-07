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
    "data"
)

AUDIO_DIR = (
    DATA_DIR /
    "audio_files"
)

METADATA_DIR = (
    DATA_DIR /
    "metadata"/
    "fma_small"
)

SPECTROGRAM_DIR = (
    DATA_DIR /
    "spectrograms"
)


# ==================================================
# FMA Dataset Paths
# ==================================================

FMA_AUDIO_DIR = (
    AUDIO_DIR /
    "source" /
    "fma_small"
)

FMA_SPECTROGRAM_DIR = (
    SPECTROGRAM_DIR /
    "fma_small"
)

FMA_INVENTORY_FILE = (
    METADATA_DIR /
    "fma_small_inventory.csv"
)

FMA_SPLIT_FILE = (
    METADATA_DIR /
    "fma_song_splits.csv"
)


# ==================================================
# Results
# ==================================================

RESULTS_DIR = (
    PROJECT_ROOT /
    "results" /
    "fma_small"
)


# ==================================================
# Analytics
# ==================================================

ANALYTICS_DIR = (
    PROJECT_ROOT /
    "analytics" /
    "fma_small"
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
    Creates and returns
    results/fma_small/<model_name>/*

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

    root = (
        RESULTS_DIR /
        model_name
    )

    paths = {

        "root":
            root,

        "checkpoints":
            root /
            "checkpoints",

        "confusion_matrices":
            root /
            "confusion_matrices",

        "duration_results":
            root /
            "duration_results",

        "genre_results":
            root /
            "genre_results",

        "histories":
            root /
            "histories",

        "models":
            root /
            "models",

        "metadata":
            root /
            "metadata"
    }

    for path in paths.values():

        path.mkdir(
            parents=True,
            exist_ok=True
        )

    return paths
