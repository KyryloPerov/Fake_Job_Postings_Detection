"""Download and load the raw Kaggle dataset.

download_data() never runs on import. Call it once from a notebook, or run
`python -m src.data` from the project root.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config


def download_data(force: bool = False) -> Path:
    """Download the dataset with kagglehub and copy the CSV into data/raw/.

    Needs Kaggle credentials: either ~/.kaggle/kaggle.json or the KAGGLE_USERNAME
    and KAGGLE_KEY environment variables. Returns the path of the local CSV.
    """
    if config.RAW_CSV_PATH.exists() and not force:
        print(f"{config.RAW_CSV_PATH} already exists, skipping download.")
        return config.RAW_CSV_PATH

    import kagglehub  # imported lazily: only needed for this one-off download

    cache_dir = Path(kagglehub.dataset_download(config.KAGGLE_DATASET))
    source = next(cache_dir.rglob(config.RAW_CSV_NAME), None)
    if source is None:
        raise FileNotFoundError(f"{config.RAW_CSV_NAME} not found under {cache_dir}")

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, config.RAW_CSV_PATH)
    print(f"Saved {config.RAW_CSV_PATH}")
    return config.RAW_CSV_PATH


def load_raw(path: Optional[Path] = None) -> pd.DataFrame:
    """Read fake_job_postings.csv into a DataFrame."""
    path = Path(path) if path is not None else config.RAW_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.")
    return pd.read_csv(path)


def drop_duplicate_postings(df: pd.DataFrame) -> pd.DataFrame:
    """Drop postings identical on every column except job_id.

    job_id is unique by construction, so de-duplicating on it would be a no-op -
    the real duplicates are the same advert posted several times.
    """
    subset = [c for c in df.columns if c != config.ID_COLUMN]
    before = len(df)
    df = df.drop_duplicates(subset=subset).reset_index(drop=True)
    print(f"Dropped {before - len(df)} duplicate postings ({before} -> {len(df)})")
    return df


if __name__ == "__main__":
    download_data()
