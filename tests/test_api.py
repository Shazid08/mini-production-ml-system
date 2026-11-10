"""FastAPI service: status codes, response contract, fallback layers, 503."""

import pytest

from payloads import REJECTED_CASES, with_changes


def test_health_reports_model_loaded_and_failure_switch_off(client, bundle):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is True
    assert body["threshold"] == bundle["threshold"]
    assert body["simulated_failure"] is None


def test_valid_request_is_scored_by_the_model(client, valid_payload, bundle):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert (body["request_id"], body["customer_id"]) == ("req-test", "CUST-TEST")
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["source"] == "model" and body["fallback_reason"] is None
    assert body["threshold"] == bundle["threshold"]
    assert body["churn_prediction"] == (body["churn_probability"] >= bundle["threshold"])
    assert body["warnings"] == []


def test_request_id_is_generated_when_missing(client, valid_payload):
    valid_payload.pop("request_id")
    body = client.post("/predict", json=valid_payload).json()
    assert len(body["request_id"]) == 36  # UUID4 string


@pytest.mark.parametrize("case,changes,field", REJECTED_CASES, ids=[c[0] for c in REJECTED_CASES])
def test_invalid_payloads_return_422(client, case, changes, field):
    response = client.post("/predict", json=with_changes(changes))
    assert response.status_code == 422
    locations = [".".join(str(p) for p in e["loc"]) for e in response.json()["detail"]]
    assert ("body." + field if field else "body") in locations


def test_broken_json_returns_422(client):
    response = client.post("/predict", content=b'{"customer_id": "X",', headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_null_total_charges_for_new_customer_is_scored_with_note(client):
    body = client.post("/predict", json=with_changes({"tenure_months": 0, "total_charges": None})).json()
    assert body["source"] == "model"
    assert any("total_charges was null" in w for w in body["warnings"])


def test_extreme_but_possible_values_are_scored_with_warnings(client):
    payload = with_changes({"tenure_months": 300, "monthly_charges": 250.0, "total_charges": 75000.0})
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert len(warnings) == 3 and all("outside the training range" in w for w in warnings)


# ---------------------------------------------------------------- fallback layers
def test_layer_2_heuristic_when_model_fails(make_client, valid_payload, bundle):
    client = make_client(CHURN_API_SIMULATE_FAILURE="model")
    body = client.post("/predict", json=valid_payload).json()
    # Valid payload: month-to-month and tenure 4 -> heuristic score 2.
    assert body["source"] == "fallback_heuristic"
    assert body["churn_probability"] == bundle["heuristic"]["churn_rate_by_score"][2]
    assert body["churn_prediction"] is True and body["threshold"] is None
    assert "Model scoring failed" in body["fallback_reason"]


def test_layer_3_prior_when_model_and_heuristic_fail(make_client, valid_payload, bundle):
    client = make_client(CHURN_API_SIMULATE_FAILURE="all")
    body = client.post("/predict", json=valid_payload).json()
    assert body["source"] == "fallback_prior_rate"
    assert body["churn_probability"] == bundle["train_churn_rate"]
    assert body["churn_prediction"] is False
    assert "Heuristic rule also failed" in body["fallback_reason"]


class _BrokenModel:
    def predict_proba(self, X):
        raise ValueError("corrupted model")


def test_fallback_also_triggers_on_a_real_model_error(client, valid_payload, bundle):
    # No simulation switch: the model object itself raises.
    client.app.state.bundle = {**bundle, "pipeline": _BrokenModel()}
    body = client.post("/predict", json=valid_payload).json()
    assert body["source"] == "fallback_heuristic"
    assert "ValueError: corrupted model" in body["fallback_reason"]


def test_prior_used_on_a_real_heuristic_error(client, valid_payload, bundle):
    broken_heuristic = {"threshold": 2.0, "churn_rate_by_score": {}}  # lookup will raise KeyError
    client.app.state.bundle = {**bundle, "pipeline": _BrokenModel(), "heuristic": broken_heuristic}
    body = client.post("/predict", json=valid_payload).json()
    assert body["source"] == "fallback_prior_rate"
    assert "KeyError" in body["fallback_reason"]


# ---------------------------------------------------------------- 503
@pytest.mark.parametrize("artifact", ["missing", "corrupt"])
def test_503_when_model_artifact_cannot_be_loaded(make_client, valid_payload, tmp_path, artifact):
    path = tmp_path / "model.joblib"
    if artifact == "corrupt":
        path.write_bytes(b"not a joblib file")
    client = make_client(CHURN_MODEL_PATH=str(path))
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["model_loaded"] is False
    assert client.post("/predict", json=valid_payload).status_code == 503
