"""Shared fixtures. Nothing here needs the Kaggle dataset; only the committed
artifact, data/sample_input.csv and tests/fixtures/ are used."""

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from payloads import VALID_PAYLOAD
from src.data_pipeline import REPO_ROOT

ARTIFACT_PATH = REPO_ROOT / "artifacts" / "churn_model.joblib"
SAMPLE_INPUT = REPO_ROOT / "data" / "sample_input.csv"
MESSY_INPUT = Path(__file__).parent / "fixtures" / "messy_input.csv"


@pytest.fixture
def valid_payload():
    return dict(VALID_PAYLOAD)


@pytest.fixture(scope="session")
def bundle():
    return joblib.load(ARTIFACT_PATH)


@pytest.fixture
def make_client(monkeypatch):
    """Start the real app (startup included) with the given environment variables."""
    from serving.app import app
    started = []

    def _make(**env):
        monkeypatch.delenv("CHURN_API_SIMULATE_FAILURE", raising=False)
        monkeypatch.delenv("CHURN_MODEL_PATH", raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        client = TestClient(app)
        client.__enter__()  # runs the lifespan, i.e. loads the model
        started.append(client)
        return client

    yield _make
    for client in started:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    """App started with default settings (failure switch off)."""
    return make_client()
