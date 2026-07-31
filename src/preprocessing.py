"""Cleaning, splitting, and fitting the transformers.

The split happens once and is written to data/processed/, so every model in the
project - local and Colab - is scored on exactly the same test rows.

Fit and transform are separate functions on purpose: preprocess_data() fits on
train, preprocess_new_data() only transforms val/test with what train produced.
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config, data, features

TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")


# --- cleaning --------------------------------------------------------------


def clean_text(text: Any) -> str:
    """Unescape HTML entities, drop tags, collapse whitespace. NaN -> ""."""
    if pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = TAG_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the text fields and normalise the categoricals. Leaves the target alone."""
    df = df.copy()
    for col in config.TEXT_COLUMNS:
        df[col] = df[col].map(clean_text)
    df[config.LOCATION_COLUMN] = df[config.LOCATION_COLUMN].map(clean_text)
    for col in config.CATEGORICAL_COLUMNS:
        cleaned = df[col].fillna("").astype(str).str.strip()
        df[col] = cleaned.replace("", "Unknown")
    return df


# --- splitting -------------------------------------------------------------


def split_data(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    val_size: float = config.VAL_SIZE,
    seed: int = config.SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split on `fraudulent`. Same seed -> same rows."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[config.TARGET], random_state=seed
    )
    # val_size is a fraction of the whole dataset, so rescale it for this call.
    relative_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[config.TARGET],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path = config.PROCESSED_DATA_DIR,
) -> None:
    """Write train.csv / val.csv / test.csv. These are the shared contract."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"{path}  {len(part):>6} rows, {part[config.TARGET].mean():.2%} fraudulent")


def load_splits(
    out_dir: Path = config.PROCESSED_DATA_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three CSVs back, restoring the empty strings that a CSV turns into NaN."""
    frames = []
    for name in ("train", "val", "test"):
        path = out_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.preprocessing` first."
            )
        df = pd.read_csv(path)
        for col in config.TEXT_COLUMNS + [config.FULL_TEXT_COLUMN]:
            df[col] = df[col].fillna("").astype(str)
        for col in features.CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("Unknown").astype(str)
        frames.append(df)
    return tuple(frames)


def build_processed_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """raw CSV -> deduplicated -> cleaned -> featurised -> split -> saved."""
    df = data.load_raw()
    df = data.drop_duplicate_postings(df)
    df = clean_dataframe(df)
    df = features.add_all_features(df)
    train, val, test = split_data(df)
    save_splits(train, val, test)
    return train, val, test


# --- transformers (fit on train, reuse on val/test) ------------------------


def build_text_vectorizer(
    texts, params: Optional[dict] = None
) -> Tuple[Any, TfidfVectorizer]:
    """Fit a TF-IDF vectorizer on the training texts. Returns (matrix, vectorizer)."""
    vectorizer = TfidfVectorizer(**(params or config.TFIDF_PARAMS))
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def encode_categorical_features(
    df: pd.DataFrame, columns, encoder: Optional[OneHotEncoder] = None
) -> Tuple[np.ndarray, OneHotEncoder]:
    """One-hot encode `columns`.

    Pass encoder=None to fit (train) or a fitted encoder to transform (val/test).
    min_frequency folds rare industries into one bucket instead of exploding the
    matrix; handle_unknown="ignore" absorbs categories only seen at test time.
    """
    values = df[columns].astype(str)
    if encoder is None:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False
        )
        matrix = encoder.fit_transform(values)
    else:
        matrix = encoder.transform(values)
    return matrix, encoder


def scale_numeric_features(
    df: pd.DataFrame, columns, scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """Standardise numeric columns. scaler=None to fit, otherwise transform."""
    values = df[columns].astype(float).to_numpy()
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(values), scaler
    return scaler.transform(values), scaler


UNSAFE_NAME_RE = re.compile(r"[^0-9a-zA-Z_]+")


def sanitize_feature_names(names) -> list:
    """Make column names safe for LightGBM, which rejects JSON-special characters.

    The one-hot names embed real category values, and this dataset's taxonomy is
    full of commas and ampersands ("Health, Wellness and Fitness"). LightGBM fails
    outright on those, so collapse anything non-alphanumeric to "_" and suffix the
    collisions that creates.
    """
    seen: Dict[str, int] = {}
    safe = []
    for name in names:
        cleaned = UNSAFE_NAME_RE.sub("_", str(name)).strip("_") or "feature"
        count = seen.get(cleaned, 0)
        seen[cleaned] = count + 1
        safe.append(cleaned if count == 0 else f"{cleaned}__{count}")
    return safe


def _feature_names(svd_components: int, encoder: OneHotEncoder) -> list:
    """Column names for the stacked matrix, so importances stay readable."""
    text = [f"svd_{i}" for i in range(svd_components)]
    categorical = list(encoder.get_feature_names_out(features.CATEGORICAL_FEATURES))
    return sanitize_feature_names(text + categorical + list(features.NUMERIC_FEATURES))


def preprocess_data(
    train_df: pd.DataFrame,
    svd_components: int = config.SVD_COMPONENTS,
    scale_numeric: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """Fit every transformer on the training split.

    Returns (X_train, y_train, artifacts). X is a dense named DataFrame stacking
    three blocks: TruncatedSVD over TF-IDF(full_text), one-hot categoricals, and
    the numeric hand-crafted features. Named columns are what let LightGBM report
    importances by feature instead of by position.
    """
    tfidf_matrix, vectorizer = build_text_vectorizer(train_df[config.FULL_TEXT_COLUMN])
    svd = TruncatedSVD(n_components=svd_components, random_state=config.SEED)
    text_block = svd.fit_transform(tfidf_matrix)

    categorical_block, encoder = encode_categorical_features(
        train_df, features.CATEGORICAL_FEATURES
    )

    scaler = None
    if scale_numeric:
        numeric_block, scaler = scale_numeric_features(train_df, features.NUMERIC_FEATURES)
    else:
        numeric_block = train_df[features.NUMERIC_FEATURES].astype(float).to_numpy()

    feature_names = _feature_names(svd_components, encoder)
    X = pd.DataFrame(
        np.hstack([text_block, categorical_block, numeric_block]),
        columns=feature_names,
        index=train_df.index,
    )
    y = train_df[config.TARGET].to_numpy()
    artifacts = {
        "vectorizer": vectorizer,
        "svd": svd,
        "encoder": encoder,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    return X, y, artifacts


def preprocess_new_data(
    df: pd.DataFrame, artifacts: Dict[str, Any]
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Transform val/test with the transformers fitted on train. Nothing is refitted."""
    tfidf_matrix = artifacts["vectorizer"].transform(df[config.FULL_TEXT_COLUMN])
    text_block = artifacts["svd"].transform(tfidf_matrix)

    categorical_block, _ = encode_categorical_features(
        df, features.CATEGORICAL_FEATURES, encoder=artifacts["encoder"]
    )

    if artifacts["scaler"] is not None:
        numeric_block, _ = scale_numeric_features(
            df, features.NUMERIC_FEATURES, scaler=artifacts["scaler"]
        )
    else:
        numeric_block = df[features.NUMERIC_FEATURES].astype(float).to_numpy()

    X = pd.DataFrame(
        np.hstack([text_block, categorical_block, numeric_block]),
        columns=artifacts["feature_names"],
        index=df.index,
    )
    y = df[config.TARGET].to_numpy() if config.TARGET in df.columns else None
    return X, y


if __name__ == "__main__":
    config.ensure_dirs()
    build_processed_dataset()
