"""Training and inference for each model in the comparison.

Every train_* function returns a fitted estimator. Use predict_proba() on it to
get the probability of the positive (fraudulent) class as a 1-D array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config


def compute_scale_pos_weight(y) -> float:
    """neg/pos ratio - what LightGBM's scale_pos_weight expects for imbalance."""
    y = np.asarray(y)
    positives = int((y == 1).sum())
    if positives == 0:
        raise ValueError("no positive samples in y")
    return float((y == 0).sum()) / positives


# --- baselines -------------------------------------------------------------


def train_most_frequent_baseline(X_train, y_train) -> DummyClassifier:
    """Always predict the majority class. The reference that shows accuracy is a lie."""
    model = DummyClassifier(strategy="most_frequent", random_state=config.SEED)
    model.fit(X_train, y_train)
    return model


def train_tfidf_logreg(
    texts,
    y_train,
    tfidf_params: Optional[dict] = None,
    logreg_params: Optional[dict] = None,
) -> Pipeline:
    """Baseline: TF-IDF over full_text -> logistic regression with balanced classes."""
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**(tfidf_params or config.TFIDF_PARAMS))),
            ("logreg", LogisticRegression(**(logreg_params or config.LOGREG_PARAMS))),
        ]
    )
    model.fit(texts, y_train)
    return model


# --- model 1: LightGBM -----------------------------------------------------


def train_lightgbm(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    params: Optional[dict] = None,
):
    """LightGBM on SVD(TF-IDF) + one-hot categoricals + hand-crafted features.

    Pass X_val/y_val to early-stop on validation average precision. scale_pos_weight
    is derived from y_train unless the caller overrides it in `params`.

    Pass X as a named DataFrame (what preprocess_data returns). LightGBM then
    reports importances by feature name, and sklearn stops warning that a bare
    numpy array "does not have valid feature names" on every predict().
    """
    import lightgbm as lgb

    params = dict(params or config.LIGHTGBM_PARAMS)
    params.setdefault("scale_pos_weight", compute_scale_pos_weight(y_train))
    model = lgb.LGBMClassifier(**params)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "average_precision",
            "callbacks": [
                lgb.early_stopping(config.LIGHTGBM_EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(0),
            ],
        }

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def feature_importance_frame(model) -> "pd.DataFrame":
    """Gain-based importances as a named, sorted frame.

    Reads the names off the booster, which has them because the model was fitted
    on a named DataFrame.
    """
    import pandas as pd

    booster = model.booster_
    frame = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
        }
    )
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def embeddings_to_frame(embeddings: np.ndarray, prefix: str = "emb") -> "pd.DataFrame":
    """Wrap an embedding matrix in a named DataFrame (emb_0 ... emb_n).

    Same reason as above: use it for both fit and predict so the column names stay
    consistent and LightGBM/sklearn have nothing to complain about.
    """
    import pandas as pd

    embeddings = np.asarray(embeddings)
    return pd.DataFrame(
        embeddings, columns=[f"{prefix}_{i}" for i in range(embeddings.shape[1])]
    )


# --- model 3: frozen sentence-transformer embeddings + classifier ----------


def encode_texts(
    texts,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    max_chars: int = config.SENTENCE_TRANSFORMER_MAX_CHARS,
) -> np.ndarray:
    """Encode texts with a frozen sentence-transformer. No fine-tuning happens here.

    Runs on CPU (slow but fine) or on a GPU if device="cuda". Cache the result -
    encoding the full dataset is the expensive part of that notebook.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name or config.SENTENCE_TRANSFORMER_MODEL, device=device)
    truncated = [str(t)[:max_chars] for t in texts]
    return model.encode(
        truncated,
        batch_size=batch_size or config.SENTENCE_TRANSFORMER_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def train_embedding_classifier(X_train, y_train, kind: str = "logreg"):
    """Classifier on top of the frozen embeddings: "logreg" or "lightgbm"."""
    if kind == "logreg":
        model = LogisticRegression(**config.EMBEDDING_LOGREG_PARAMS)
        model.fit(X_train, y_train)
        return model
    if kind == "lightgbm":
        return train_lightgbm(X_train, y_train)
    raise ValueError(f"kind must be 'logreg' or 'lightgbm', got {kind!r}")


# --- inference and persistence ---------------------------------------------


def predict_proba(model, X) -> np.ndarray:
    """Probability of the positive class as a 1-D array."""
    return model.predict_proba(X)[:, 1]


def save_model(model: Any, name: str) -> Path:
    """Pickle a fitted model into models/<name>.joblib."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved {path}")
    return path


def load_model(name: str) -> Any:
    """Load a model saved by save_model()."""
    return joblib.load(config.MODELS_DIR / f"{name}.joblib")


def save_test_predictions(name: str, y_true, y_proba) -> Path:
    """Store test-set probabilities so every model can share one comparison plot.

    The Colab DistilBERT notebook writes a file in this same format.
    """
    import pandas as pd

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.MODELS_DIR / f"predictions_{name}_test.csv"
    pd.DataFrame({"y_true": np.asarray(y_true), "y_proba": np.asarray(y_proba)}).to_csv(
        path, index=False
    )
    print(f"Saved {path}")
    return path
