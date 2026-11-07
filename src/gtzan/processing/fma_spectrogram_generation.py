from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================================================
# Configuration
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

AUDIO_DIR = (
    PROJECT_DIR
    / "data"
    / "audio_files"
    / "source"
    / "fma_small"
)

SPECTROGRAM_DIR = (
    PROJECT_DIR
    / "data"
    / "spectrograms"
    / "fma_small"
)

INVENTORY_FILE = (
    PROJECT_DIR
    / "data"
    / "metadata"
    / "fma_small_inventory.csv"
)

DURATIONS = [
    #1,
    #3,
    #5,
    #10,
    #15,
    #20,
    #25,
    30
]


# ==================================================
# Spectrogram Creation
# ==================================================

def save_spectrogram(
    audio_path,
    output_path,
    duration_seconds
):
    """
    Generate a Mel spectrogram from the first
    N seconds of an audio file.
    """

    try:

        y, sr = librosa.load(
            audio_path,
            sr=None,
            duration=duration_seconds
        )

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        fig = plt.figure(
            figsize=(3, 3),
            frameon=False
        )

        ax = plt.Axes(
            fig,
            [0, 0, 1, 1]
        )

        ax.set_axis_off()

        fig.add_axes(ax)

        librosa.display.specshow(
            mel_db,
            sr=sr,
            cmap="gray"
        )

        fig.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(fig)

        return True

    except Exception as e:

        print(
            f"ERROR creating spectrogram: "
            f"{audio_path}"
        )

        print(e)

        return False


# ==================================================
# Main Processing
# ==================================================

def main():

    inventory_df = pd.read_csv(
        INVENTORY_FILE
    )

    print(
        f"Loaded "
        f"{len(inventory_df):,} tracks"
    )

    total_generated = 0

    for duration in DURATIONS:

        duration_label = (
            f"{duration}s"
        )

        print(
            f"\nProcessing "
            f"{duration_label}"
        )

        duration_count = 0

        for row in inventory_df.itertuples():

            audio_path = (
                AUDIO_DIR /
                row.relative_path
            )

            genre = row.genre.lower()

            output_dir = (
                SPECTROGRAM_DIR /
                duration_label /
                genre
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            output_file = (
                output_dir /
                f"{row.track_id}.png"
            )

            if output_file.exists():
                continue

            success = save_spectrogram(
                audio_path=audio_path,
                output_path=output_file,
                duration_seconds=duration
            )

            if success:

                total_generated += 1
                duration_count += 1

                if total_generated % 100 == 0:

                    print(
                        f"Generated "
                        f"{total_generated:,} "
                        f"spectrograms"
                    )

        print(
            f"Completed {duration_label}: "
            f"{duration_count:,} files"
        )

    print("\nDone.")

    print(
        f"Total spectrograms generated: "
        f"{total_generated:,}"
    )


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":
    main()