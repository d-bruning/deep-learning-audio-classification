from pathlib import Path
import librosa
import soundfile as sf

INPUT_DIR = Path("data/audio_files/source")
OUTPUT_DIR = Path("data/audio_files/processed")


# Durations to generate (seconds)
DURATIONS = [30, 25, 20, 15, 10, 5, 3, 1]


# Preserve original sample rate
TARGET_SR = None


def extract_center_clip(audio, sr, duration):
    """
    Extract a centered audio clip of specified duration.
    """
    target_samples = int(duration * sr)

    if len(audio) <= target_samples:
        return audio

    start = (len(audio) - target_samples) // 2
    end = start + target_samples

    return audio[start:end]


def process_file(input_file, genre):
    try:
        audio, sr = librosa.load(
            input_file,
            sr=TARGET_SR,
            mono=True
        )

        for duration in DURATIONS:
            clip = extract_center_clip(audio, sr, duration)

            out_dir = OUTPUT_DIR / f"{duration}s" / genre
            out_dir.mkdir(parents=True, exist_ok=True)

            output_file = out_dir / input_file.name

            sf.write(output_file, clip, sr)

    except Exception as e:
        print(f"ERROR: {input_file} -> {e}")


def main():
    genres = [g for g in SOURCE_DIR.iterdir() if g.is_dir()]

    total_files = 0

    for genre_dir in genres:
        genre = genre_dir.name

        wav_files = list(genre_dir.glob("*.wav"))

        print(f"Processing {genre}: {len(wav_files)} files")

        for wav_file in wav_files:
            process_file(wav_file, genre)
            total_files += 1

    print()
    print(f"Processed {total_files} source files")
    print(
        f"Generated approximately "
        f"{total_files * len(DURATIONS)} clips"
    )


if __name__ == "__main__":
    main()
