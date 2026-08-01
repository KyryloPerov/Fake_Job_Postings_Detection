"""Hand-crafted features built from the raw posting fields.

Three families:
  * missingness flags - a fake posting often has no company profile, no logo and
    no salary range, so the absence itself carries signal;
  * cheap text statistics - length, shouting (CAPS), exclamation marks;
  * contact details - whether an email / URL / phone appears in the text.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src import config

# The Kaggle export masks contact details as #EMAIL_<hash>#, #URL_<hash># and
# #PHONE_<hash>#, so match both the placeholders and any plain occurrences.
EMAIL_RE = re.compile(r"#EMAIL_[0-9a-f]+#|[\w.+-]+@[\w-]+\.[\w.]{2,}", re.IGNORECASE)
URL_RE = re.compile(r"#URL_[0-9a-f]+#|https?://\S+|www\.\S+", re.IGNORECASE)
PHONE_RE = re.compile(r"#PHONE_[0-9a-f]+#|\+?\d[\d\s().-]{8,}\d")
MONEY_RE = re.compile(r"[$€£]\s?\d|\b\d{2,}\s?k\b|\bper hour\b", re.IGNORECASE)

MISSING_INDICATOR_FEATURES = [f"{c}_is_missing" for c in config.MISSINGNESS_COLUMNS]
LENGTH_FEATURES = [
    f"{c}_{stat}"
    for c in config.TEXT_COLUMNS + [config.FULL_TEXT_COLUMN]
    for stat in ("char_count", "word_count")
]
PATTERN_FEATURES = [
    "caps_ratio",
    "digit_ratio",
    "exclamation_count",
    "question_count",
    "has_email",
    "has_url",
    "has_phone",
    "has_money",
]
LOCATION_FEATURES = ["location_depth"]
SALARY_FEATURES = ["salary_low", "salary_high", "salary_span"]

# Everything the models consume, in a stable order.
NUMERIC_FEATURES = (
    config.BINARY_COLUMNS
    + MISSING_INDICATOR_FEATURES
    + LENGTH_FEATURES
    + PATTERN_FEATURES
    + LOCATION_FEATURES
    + SALARY_FEATURES
)
CATEGORICAL_FEATURES = config.CATEGORICAL_COLUMNS + config.DERIVED_CATEGORICAL_COLUMNS


def _is_missing(series: pd.Series) -> pd.Series:
    """True for NaN and for empty / whitespace-only strings.

    Robust to being called before or after the text columns are filled with "".
    """
    return series.isna() | (series.astype(str).str.strip() == "")


def build_full_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate the text fields into one string per posting (missing -> "")."""
    parts = df[config.TEXT_COLUMNS].fillna("").astype(str)
    joined = parts.agg(" ".join, axis=1)
    return joined.str.replace(r"\s+", " ", regex=True).str.strip()


def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """One 0/1 column per field whose absence might be informative."""
    df = df.copy()
    for col in config.MISSINGNESS_COLUMNS:
        df[f"{col}_is_missing"] = _is_missing(df[col]).astype(int)
    return df


def add_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Character and word counts for every text field and for full_text."""
    df = df.copy()
    for col in config.TEXT_COLUMNS + [config.FULL_TEXT_COLUMN]:
        text = df[col].fillna("").astype(str)
        df[f"{col}_char_count"] = text.str.len()
        df[f"{col}_word_count"] = text.str.split().str.len().fillna(0).astype(int)
    return df


def add_text_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """Shouting, punctuation and contact details, measured on full_text."""
    df = df.copy()
    text = df[config.FULL_TEXT_COLUMN].fillna("").astype(str)

    letters = text.str.count(r"[A-Za-z]")
    df["caps_ratio"] = (text.str.count(r"[A-Z]") / letters.replace(0, np.nan)).fillna(0.0)
    df["digit_ratio"] = (
        text.str.count(r"\d") / text.str.len().replace(0, np.nan)
    ).fillna(0.0)
    df["exclamation_count"] = text.str.count("!")
    df["question_count"] = text.str.count(r"\?")

    df["has_email"] = text.str.contains(EMAIL_RE, na=False).astype(int)
    df["has_url"] = text.str.contains(URL_RE, na=False).astype(int)
    df["has_phone"] = text.str.contains(PHONE_RE, na=False).astype(int)
    df["has_money"] = text.str.contains(MONEY_RE, na=False).astype(int)
    return df


def add_location_features(df: pd.DataFrame) -> pd.DataFrame:
    """Split `location` ("US, NY, New York") into a country and a detail level.

    location_depth counts the non-empty comma-separated parts: vague locations
    such as "US, , " score lower than a fully specified city.
    """
    df = df.copy()
    parts = df[config.LOCATION_COLUMN].fillna("").astype(str).str.split(",")
    country = parts.str[0].str.strip().str.upper()
    df["country"] = country.replace("", "Unknown")
    df["location_depth"] = parts.apply(lambda ps: sum(1 for p in ps if p.strip()))
    return df


def add_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the free-form `salary_range` ("50000-60000") into low / high / span.

    Present for only ~16% of postings and the units are inconsistent (yearly,
    hourly, other currencies), so treat the numbers as weak hints. Unparseable
    values become -1: trees split on it happily and the scaler tolerates it,
    whereas NaN would need a separate imputation step.
    """
    df = df.copy()
    raw = df[config.SALARY_COLUMN].fillna("").astype(str)
    bounds = raw.str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    low = pd.to_numeric(bounds[0], errors="coerce")
    high = pd.to_numeric(bounds[1], errors="coerce")

    df["salary_low"] = low.fillna(-1)
    df["salary_high"] = high.fillna(-1)
    df["salary_span"] = (high - low).fillna(-1)
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature function in order. full_text must exist before the rest."""
    df = df.copy()
    df[config.FULL_TEXT_COLUMN] = build_full_text(df)
    df = add_missing_indicators(df)
    df = add_text_length_features(df)
    df = add_text_pattern_features(df)
    df = add_location_features(df)
    df = add_salary_features(df)
    return df
