"""Data pipeline: split logic, cleaning rules, and checks on the real data.

Tests marked with the `processed_splits` / `raw_file` fixtures need the Kaggle
dataset and SKIP (with a message) on a fresh clone that has not downloaded it.
"""

import numpy as np
import pandas as pd
import pytest

from serving.schemas import PredictRequest
from src.data_pipeline import (ALLOWED_CATEGORIES, FEATURES, ID_COLUMN, PROCESSED_DIR, RAW_DIR, RAW_FILENAME,
                               check_categories, clean, fill_new_customer_total_charges, load_raw, split_data)

NEEDS_DATA = "run `python -m src.data_pipeline` first (downloads the dataset from Kaggle)"


@pytest.fixture(scope="module")
def processed_splits():
    paths = {name: PROCESSED_DIR / f"{name}.csv" for name in ("train", "val", "test")}
    missing = [str(p.name) for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"processed splits not found ({', '.join(missing)}); {NEEDS_DATA}")
    return {name: pd.read_csv(p, dtype={ID_COLUMN: str}) for name, p in paths.items()}


@pytest.fixture(scope="module")
def raw_file():
    path = RAW_DIR / RAW_FILENAME
    if not path.exists():
        pytest.skip(f"raw dataset {path.name} not found; {NEEDS_DATA}")
    return load_raw(path)


def test_split_is_70_15_15_stratified_and_disjoint():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"id": range(1000), "y": (rng.random(1000) < 0.265).astype(int)})
    train, val, test = split_data(df, target="y")
    assert (len(train), len(val), len(test)) == (700, 150, 150)
    ids = [set(part["id"]) for part in (train, val, test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set().union(*ids) == set(df["id"])
    for part in (train, val, test):
        assert abs(part["y"].mean() - df["y"].mean()) < 0.01


def test_split_is_reproducible():
    df = pd.DataFrame({"id": range(200), "y": [0, 1] * 100})
    first, second = split_data(df, target="y"), split_data(df, target="y")
    assert all(a["id"].tolist() == b["id"].tolist() for a, b in zip(first, second))


def test_total_charges_rule_only_fills_new_customers():
    total = pd.Series([np.nan, np.nan, 100.0])
    tenure = pd.Series([0, 5, 3])
    result = fill_new_customer_total_charges(total, tenure)
    assert result[0] == 0.0 and np.isnan(result[1]) and result[2] == 100.0


def test_unknown_category_in_training_data_is_refused():
    df = pd.DataFrame({f: [values[0]] for f, values in ALLOWED_CATEGORIES.items()})
    check_categories(df)  # allowed values pass
    df.loc[0, "contract"] = "Three year"
    with pytest.raises(ValueError, match="contract"):
        check_categories(df)


# ---------------------------------------------------------------- need the dataset (skip on a fresh clone)
def test_no_customer_appears_in_more_than_one_split(processed_splits):
    ids = {name: set(df[ID_COLUMN]) for name, df in processed_splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert sum(len(v) for v in ids.values()) == 7043


def test_every_processed_row_passes_the_api_schema(processed_splits):
    for df in processed_splits.values():
        for record in df.drop(columns="churn").to_dict(orient="records"):
            PredictRequest.model_validate(record)


def test_cleaning_raw_file_removes_leak_columns_and_blanks(raw_file):
    cleaned = clean(raw_file)
    assert list(cleaned.columns) == [ID_COLUMN] + FEATURES + ["churn"]
    assert cleaned.isna().sum().sum() == 0
    assert (cleaned.loc[cleaned["tenure_months"] == 0, "total_charges"] == 0.0).all()
