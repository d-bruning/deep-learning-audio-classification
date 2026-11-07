import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))

from utils.paths import (
    SPECTROGRAM_DIR,
    SPLIT_FILE
)

import pandas as pd

from sklearn.model_selection import train_test_split

print(f"SPECTROGRAM_DIR: {SPECTROGRAM_DIR}")
print(f"SPLIT_FILE: {SPLIT_FILE}")

source_dir = (
    SPECTROGRAM_DIR /
    "processed" /
    "30s"
)

print(f"Source directory: {source_dir}")
print(f"Exists: {source_dir.exists()}")

def create_song_split_file():

    rows = []

    source_dir = (
        SPECTROGRAM_DIR /
        "processed" /
        "30s"
    )

    for genre_dir in source_dir.iterdir():

        genre = genre_dir.name

        files = [
            f.stem
            for f in genre_dir.glob(
                "*.png"
            )
        ]

        train_files, temp_files = (
            train_test_split(
                files,
                test_size=0.30,
                random_state=42
            )
        )

        val_files, test_files = (
            train_test_split(
                temp_files,
                test_size=0.50,
                random_state=42
            )
        )

        for f in train_files:
            rows.append(
                [genre, f, "train"]
            )

        for f in val_files:
            rows.append(
                [genre, f, "val"]
            )

        for f in test_files:
            rows.append(
                [genre, f, "test"]
            )

    split_df = pd.DataFrame(
        rows,
        columns=[
            "genre",
            "file_id",
            "split"
        ]
    )

    split_df.to_csv(
        SPLIT_FILE,
        index=False
    )

if SPLIT_FILE.exists():
    print("song_splits.csv already exists")
else:
    create_song_split_file()