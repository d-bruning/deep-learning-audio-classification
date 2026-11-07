from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

INPUT_DIR = Path("data/audio_files/processed")
OUTPUT_DIR = Path("data/spectrograms/processed")

DURATIONS_ORG = [
    "30s",
    "15s",
    "10s",
    "5s",
    "3s",
    "1s"
]

DURATIONS = [
    "25s",
    "20s"
]


def save_spectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=None)

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

    ax = plt.Axes(fig, [0, 0, 1, 1])
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


def main():

    total = 0

    for duration in DURATIONS:

        duration_dir = INPUT_DIR / duration

        print(f"\nProcessing {duration}")

        for genre_dir in duration_dir.iterdir():

            if not genre_dir.is_dir():
                continue

            output_genre_dir = (
                OUTPUT_DIR /
                duration /
                genre_dir.name
            )

            output_genre_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for wav_file in genre_dir.glob("*.wav"):

                output_file = (
                    output_genre_dir /
                    f"{wav_file.stem}.png"
                )

                save_spectrogram(
                    wav_file,
                    output_file
                )

                total += 1

                if total % 100 == 0:
                    print(
                        f"Generated {total} spectrograms"
                    )

    print(f"\nDone. Total spectrograms: {total}")


if __name__ == "__main__":
    main()
