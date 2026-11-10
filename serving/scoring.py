"""Inference logic shared by the API (serving/app.py) and the batch CLI (predict_batch.py).

Both entry points call the same three functions, so the API and the batch path
cannot drift apart:
  load_model(path)              load the artifact once and check it matches this code
  prepare_features(records, b)  apply the training cleaning rule, collect row warnings
  score_frame(frame, b)         vectorised scoring with a two-layer fallback:
                                model -> heuristic rule -> training churn rate
"""

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.data_pipeline import (ALLOWED_CATEGORIES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                               fill_new_customer_total_charges)
from src.train import heuristic_score

REQUIRED_KEYS = {"pipeline", "threshold", "features", "train_churn_rate", "numeric_train_ranges", "heuristic"}

# TEST-ONLY failure switch (environment variable). Unset = off, the default.
#   "model" -> the model layer raises, so the heuristic layer answers
#   "all"   -> the model and heuristic layers raise, so the prior layer answers
SIMULATE_FAILURE_ENV = "CHURN_API_SIMULATE_FAILURE"
SIMULATE_FAILURE_VALUES = {"model", "all"}

logger = logging.getLogger("churn_scoring")


class ModelLoadError(RuntimeError):
    pass


def load_model(path: Path) -> dict:
    """Load the artifact and check it matches this code; raise ModelLoadError otherwise."""
    try:
        bundle = joblib.load(path)
        missing = REQUIRED_KEYS - set(bundle)
        if missing:
            raise ValueError(f"artifact is missing keys {sorted(missing)}")
        if list(bundle["features"]) != FEATURES:
            raise ValueError("artifact feature list differs from src.data_pipeline.FEATURES")
        encoder = bundle["pipeline"].named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
        for feature, learned in zip(CATEGORICAL_FEATURES, encoder.categories_):
            if set(learned) != set(ALLOWED_CATEGORIES[feature]):
                raise ValueError(f"encoder categories for '{feature}' differ from ALLOWED_CATEGORIES")
        if not 0.0 <= bundle["threshold"] <= 1.0:
            raise ValueError(f"threshold {bundle['threshold']} is not a probability")
        return bundle
    except Exception as exc:
        raise ModelLoadError(f"{type(exc).__name__}: {exc}") from exc


def prepare_features(records: list[dict], bundle: dict):
    """Turn validated records into the model's input frame.

    Applies the same tenure-0 rule as training, and returns one list of
    warnings per row: null total_charges, values outside the training range,
    and categorical values never seen in training (possible only in batch).
    """
    frame = pd.DataFrame(records, columns=FEATURES)
    row_warnings = [[] for _ in range(len(frame))]

    def warn(mask, message_for_row):
        for i in np.flatnonzero(np.asarray(mask)):
            row_warnings[i].append(message_for_row(i))

    total = pd.to_numeric(frame["total_charges"], errors="coerce")  # None -> NaN
    warn(total.isna(), lambda i: "total_charges was null for a tenure-0 customer; treated as 0.0, as in training")
    frame["total_charges"] = fill_new_customer_total_charges(total, frame["tenure_months"]).astype(float)

    for feature in NUMERIC_FEATURES:
        low, high = bundle["numeric_train_ranges"][feature]
        values = frame[feature].astype(float)
        warn((values < low) | (values > high),
             lambda i, f=feature, v=values, lo=low, hi=high:
             f"{f}={v.iat[i]:g} is outside the training range [{lo:g}, {hi:g}]; the prediction is an extrapolation")

    for feature in CATEGORICAL_FEATURES:
        warn(~frame[feature].isin(ALLOWED_CATEGORIES[feature]),
             lambda i, f=feature: f"{f}={frame[f].iat[i]!r} was not seen in training; "
                                  "scored with this field treated as unknown")
    return frame, row_warnings


@dataclass
class ScoreResult:
    probability: np.ndarray
    prediction: np.ndarray
    threshold: Optional[float]
    decision_rule: str
    source: str
    fallback_reason: Optional[str]


def score_frame(frame: pd.DataFrame, bundle: dict, simulate_failure: Optional[str] = None) -> ScoreResult:
    """Score every row at once. Layers: model -> heuristic rule -> training churn rate."""
    threshold = float(bundle["threshold"])

    # Layer 1: the model.
    try:
        if simulate_failure in SIMULATE_FAILURE_VALUES:
            raise RuntimeError(f"simulated model failure ({SIMULATE_FAILURE_ENV}={simulate_failure})")
        with warnings.catch_warnings():
            # Unseen categories are already reported per row by prepare_features.
            warnings.filterwarnings("ignore", message="Found unknown categories", category=UserWarning)
            probability = bundle["pipeline"].predict_proba(frame[FEATURES])[:, 1]
        if not np.all(np.isfinite(probability)):
            raise ValueError("model returned non-finite probabilities")
        return ScoreResult(probability, probability >= threshold, threshold,
                           f"model probability >= {threshold:.4f}", "model", None)
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
        logger.error("Model layer failed for %d row(s): %s", len(frame), model_error,
                     exc_info=simulate_failure is None)

    # Layer 2: the heuristic rule from the baselines.
    heuristic = bundle["heuristic"]
    try:
        if simulate_failure == "all":
            raise RuntimeError(f"simulated heuristic failure ({SIMULATE_FAILURE_ENV}=all)")
        score = heuristic_score(frame).to_numpy()
        probability = np.array([heuristic["churn_rate_by_score"][int(s)] for s in score], dtype=float)
        reason = (f"Model scoring failed ({model_error}). Used the heuristic rule instead; churn_probability "
                  "is the training churn rate of customers with the same rule score.")
        return ScoreResult(probability, score >= heuristic["threshold"], None,
                           f"heuristic score >= {heuristic['threshold']:g} "
                           "(one point each: month-to-month contract, tenure <= 12 months)",
                           "fallback_heuristic", reason)
    except Exception as exc:
        heuristic_error = f"{type(exc).__name__}: {exc}"
        logger.error("Heuristic layer failed for %d row(s): %s", len(frame), heuristic_error,
                     exc_info=simulate_failure is None)

    # Layer 3: the training churn rate (cannot fail).
    prior = float(bundle["train_churn_rate"])
    reason = (f"Model scoring failed ({model_error}). Heuristic rule also failed ({heuristic_error}). "
              f"Returned the training churn rate {prior:.4f} instead of a customer-specific score.")
    return ScoreResult(np.full(len(frame), prior), np.full(len(frame), prior >= threshold), threshold,
                       f"training churn rate {prior:.4f} compared with model threshold {threshold:.4f}",
                       "fallback_prior_rate", reason)
