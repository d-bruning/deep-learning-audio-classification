from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio_files"
SPECTROGRAM_DIR = DATA_DIR / "spectrograms"
ANALYTICS_DIR = PROJECT_ROOT / "analytics"
SPLIT_FILE = (
    PROJECT_ROOT /
    "data" /
    "metadata" /
    "song_splits.csv"
)
FMA_INVENTORY_FILE = (
    PROJECT_ROOT /
    "data" /
    "metadata" /
    "fma_small_inventory.csv"
)
FMA_SPLIT_FILE = (
    PROJECT_ROOT /
    "data" /
    "metadata" /
    "fma_song_splits.csv"
)

def get_output_paths(model_name):

    root = RESULTS_DIR / model_name

    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "confusion_matrices": root / "confusion_matrices",
        "duration_results": root / "duration_results",
        "genre_results": root / "genre_results",
        "histories": root / "histories",
        "models": root / "models",
        "metadata": root / "metadata"
    }

    for p in paths.values():
        p.mkdir(
            parents=True,
            exist_ok=True
        )

    return paths