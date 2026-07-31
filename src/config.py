"""Seed, paths, column groups and model hyper-parameters.

Anything referenced in more than one place lives here, so the notebooks and the
modules in src/ can never disagree about a path or a parameter.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 42

# --- paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_CSV = REPORTS_DIR / "experiments.csv"

TRAIN_CSV = PROCESSED_DATA_DIR / "train.csv"
VAL_CSV = PROCESSED_DATA_DIR / "val.csv"
TEST_CSV = PROCESSED_DATA_DIR / "test.csv"

# --- data source -----------------------------------------------------------

KAGGLE_DATASET = "shivamb/real-or-fake-fake-jobposting-prediction"
RAW_CSV_NAME = "fake_job_postings.csv"
RAW_CSV_PATH = RAW_DATA_DIR / RAW_CSV_NAME

# --- columns ---------------------------------------------------------------

TARGET = "fraudulent"
ID_COLUMN = "job_id"

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
FULL_TEXT_COLUMN = "full_text"

# Low-cardinality categoricals that arrive in the raw CSV.
CATEGORICAL_COLUMNS = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]
# Categoricals produced by src/features.py.
DERIVED_CATEGORICAL_COLUMNS = ["country"]

# Already 0/1 in the raw CSV.
BINARY_COLUMNS = ["telecommuting", "has_company_logo", "has_questions"]

LOCATION_COLUMN = "location"
SALARY_COLUMN = "salary_range"
DEPARTMENT_COLUMN = "department"

# Columns whose absence is itself a signal (title is never missing, so it is out).
MISSINGNESS_COLUMNS = [
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "location",
    "salary_range",
    "department",
]

# --- splits ----------------------------------------------------------------

TEST_SIZE = 0.15  # fraction of the whole dataset
VAL_SIZE = 0.15  # fraction of the whole dataset; the rest is train

# --- model hyper-parameters ------------------------------------------------

TFIDF_PARAMS = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 3,
    "max_df": 0.9,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "stop_words": "english",
}

SVD_COMPONENTS = 200

# For TF-IDF (sparse, high-dimensional).
LOGREG_PARAMS = {
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": SEED,
}

# For sentence-transformer embeddings (dense, 384-dimensional).
EMBEDDING_LOGREG_PARAMS = {
    "C": 1.0,
    "max_iter": 2000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": SEED,
}

LIGHTGBM_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}
LIGHTGBM_EARLY_STOPPING_ROUNDS = 100

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SENTENCE_TRANSFORMER_BATCH_SIZE = 64
# The model truncates at 256 word-pieces anyway; cutting the string first only
# saves tokenizer time.
SENTENCE_TRANSFORMER_MAX_CHARS = 2000

DISTILBERT_MODEL = "distilbert-base-uncased"
DISTILBERT_PARAMS = {
    "max_length": 256,
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
}

# Optional LLM baseline (src/llm_baseline.py) - not part of the 3 graded models.
LLM_MODEL = "claude-opus-4-8"
LLM_FEW_SHOT_PER_CLASS = 4
LLM_MAX_CHARS_PER_POSTING = 3000


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch. Call once at the top of a notebook."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    """Create the output directories that are gitignored and may not exist yet."""
    for directory in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
