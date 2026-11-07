import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

SRC_DIR = (
    PROJECT_ROOT /
    "src"
)

sys.path.append(
    str(SRC_DIR)
)

from utils.paths import (
    FMA_INVENTORY_FILE,
    FMA_SPLIT_FILE
)


def create_split_file():

    inventory_df = pd.read_csv(
        FMA_INVENTORY_FILE
    )

    train_df, temp_df = train_test_split(
        inventory_df,
        test_size=0.30,
        stratify=inventory_df["genre"],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["genre"],
        random_state=42
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat(
        [
            train_df,
            val_df,
            test_df
        ],
        ignore_index=True
    )

    split_df.to_csv(
        FMA_SPLIT_FILE,
        index=False
    )

    print(
        f"Created {FMA_SPLIT_FILE}"
    )


if FMA_SPLIT_FILE.exists():
    print(
        "split file already exists"
    )
else:
    create_split_file()