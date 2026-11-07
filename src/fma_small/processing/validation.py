from pathlib import Path
import librosa
import pandas as pd

BASE_DIR = Path("data/audio_files/processed")

EXPECTED_DURATIONS = {
    "30s": 30,
    "15s": 15,
    "10s": 10,
    "5s": 5,
    "3s": 3,
    "1s": 1
}

GENRES = [
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

TOLERANCE_SECONDS = 0.05

summary_rows = []
problems = []

print("=" * 80)
print("DATASET VALIDATION")
print("=" * 80)

for duration_folder, expected_duration in EXPECTED_DURATIONS.items():

    print(f"\nChecking {duration_folder} dataset...")
    dataset_total = 0

    duration_path = BASE_DIR / duration_folder

    if not duration_path.exists():
        problems.append(
            f"Missing duration folder: {duration_folder}"
        )
        continue

    for genre in GENRES:

        genre_path = duration_path / genre

        if not genre_path.exists():
            problems.append(
                f"Missing genre folder: {duration_folder}/{genre}"
            )
            continue

        wav_files = sorted(genre_path.glob("*.wav"))

        count = len(wav_files)
        dataset_total += count

        print(f"  {genre:<10} {count:>3}")

        summary_rows.append({
            "duration_dataset": duration_folder,
            "genre": genre,
            "file_count": count
        })

        for wav_file in wav_files:
            try:
                audio, sr = librosa.load(
                    wav_file,
                    sr=None,
                    mono=True
                )

                actual_duration = len(audio) / sr

                if abs(actual_duration - expected_duration) > TOLERANCE_SECONDS:
                    problems.append(
                        f"Duration mismatch: "
                        f"{wav_file} "
                        f"(expected={expected_duration:.2f}s, "
                        f"actual={actual_duration:.2f}s)"
                    )

            except Exception as e:
                problems.append(
                    f"Unable to read file: {wav_file} "
                    f"({type(e).__name__}: {e})"
                )

    print(f"\nTotal files in {duration_folder}: {dataset_total}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

df = pd.DataFrame(summary_rows)

pivot = df.pivot(
    index="genre",
    columns="duration_dataset",
    values="file_count"
)

print(pivot)

print("\n" + "=" * 80)
print("PROBLEMS")
print("=" * 80)

if not problems:
    print("No problems detected.")
else:
    for p in problems:
        print(p)

report_file = "dataset_validation_report.csv"
df.to_csv(report_file, index=False)

print(f"\nValidation report written to: {report_file}")
