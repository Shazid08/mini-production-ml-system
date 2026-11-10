"""Inference integrity: the artifact, its feature list, and API/batch agreement."""

import numpy as np
import pandas as pd

from conftest import SAMPLE_INPUT
from predict_batch import run as run_batch
from src.data_pipeline import (ALLOWED_CATEGORIES, CATEGORICAL_FEATURES, DROPPED_COLUMNS, FEATURES, ID_COLUMN,
                               RENAME_MAP, TARGET)

LEAK_AND_ID_NAMES = {"churn", "churn_value", "churn_label", "churn_score", "cltv", "churn_reason", "customer_id",
                     "zip_code", "city", "lat_long", "latitude", "longitude", "count", "country", "state"}


def test_artifact_reproduces_its_saved_reference_predictions(bundle):
    reference = bundle["reference_predictions"]
    inputs = pd.DataFrame(reference["inputs"])[FEATURES]
    np.testing.assert_allclose(bundle["pipeline"].predict_proba(inputs)[:, 1], reference["probabilities"],
                               rtol=0, atol=1e-9)


def test_probabilities_are_between_0_and_1(bundle):
    frames = [pd.DataFrame(bundle["reference_predictions"]["inputs"]), pd.read_csv(SAMPLE_INPUT)]
    for frame in frames:
        p = bundle["pipeline"].predict_proba(frame[FEATURES])[:, 1]
        assert np.all((p >= 0) & (p <= 1))


def test_leak_and_identifier_columns_are_not_features(bundle):
    assert bundle["features"] == FEATURES
    assert not LEAK_AND_ID_NAMES & set(FEATURES)
    assert ID_COLUMN not in FEATURES and TARGET not in FEATURES
    # No audited column survives under its renamed name either.
    assert not {RENAME_MAP.get(raw) for raw in DROPPED_COLUMNS} & set(FEATURES)
    # And the fitted model itself only sees columns derived from the 19 features.
    encoded = bundle["pipeline"].named_steps["preprocess"].get_feature_names_out()
    sources = {name.split("__", 1)[1] for name in encoded}
    assert all(any(s == f or s.startswith(f + "_") for f in FEATURES) for s in sources)


def test_encoder_categories_match_allowed_categories(bundle):
    encoder = bundle["pipeline"].named_steps["preprocess"].named_transformers_["cat"].named_steps["onehot"]
    for feature, learned in zip(CATEGORICAL_FEATURES, encoder.categories_):
        assert set(learned) == set(ALLOWED_CATEGORIES[feature])


def test_api_and_batch_give_identical_output_for_the_same_rows(client, tmp_path, bundle):
    output = tmp_path / "out.csv"
    assert run_batch(SAMPLE_INPUT, output) == 0
    batch = pd.read_csv(output, dtype={"customer_id": str}).set_index("customer_id")
    for record in pd.read_csv(SAMPLE_INPUT, dtype={"customer_id": str}).to_dict(orient="records"):
        api = client.post("/predict", json=record).json()
        row = batch.loc[record["customer_id"]]
        assert api["churn_probability"] == row["churn_probability"]  # exact, no tolerance
        assert api["churn_prediction"] == bool(row["churn_prediction"])
        assert api["churn_prediction"] == (api["churn_probability"] >= bundle["threshold"])
        assert api["source"] == row["source"] == "model"
