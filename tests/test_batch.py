"""Batch CLI (predict_batch.py): outputs, row-level handling and exit codes."""

import logging

import pandas as pd
import pytest

from conftest import MESSY_INPUT, SAMPLE_INPUT
from predict_batch import main, run


def test_sample_file_is_fully_scored_and_keeps_record_ids(tmp_path, bundle):
    output = tmp_path / "out.csv"
    assert run(SAMPLE_INPUT, output) == 0
    out = pd.read_csv(output, dtype={"customer_id": str})
    expected_ids = pd.read_csv(SAMPLE_INPUT, dtype={"customer_id": str})["customer_id"]
    assert list(out["customer_id"]) == list(expected_ids)
    assert (out["status"] == "scored").all()
    assert out["churn_probability"].between(0, 1).all()
    assert (out["churn_prediction"] == (out["churn_probability"] >= bundle["threshold"])).all()


# Expected outcome for each row of tests/fixtures/messy_input.csv (by row number).
MESSY_EXPECTED = {1: "scored", 2: "scored", 3: "scored", 4: "scored", 5: "rejected", 6: "rejected",
                  7: "rejected", 8: "scored", 9: "rejected", 10: "scored", 11: "rejected", 12: "rejected",
                  13: "rejected", 14: "rejected", 15: "scored"}


def test_messy_file_rows_are_handled_individually(tmp_path, caplog):
    output = tmp_path / "out.csv"
    with caplog.at_level(logging.INFO):
        assert run(MESSY_INPUT, output) == 0
    out = pd.read_csv(output, keep_default_na=False).set_index("row_number")

    assert out["status"].to_dict() == MESSY_EXPECTED
    rejected, scored = out[out.status == "rejected"], out[out.status == "scored"]
    assert (rejected["reason"] != "").all() and (rejected["churn_probability"] == "").all()
    assert scored["churn_probability"].astype(float).between(0, 1).all()
    assert "Three year" in out.loc[3, "warnings"]                      # unseen category: warned, scored
    assert "PayPal" in out.loc[4, "warnings"]
    assert out.loc[2, "customer_id"] == "M-02"                         # padded ID trimmed
    assert "total_charges may be null only when tenure_months is 0" in out.loc[9, "reason"]
    unknown_column_logs = [r for r in caplog.records if "unknown column" in r.getMessage()]
    assert len(unknown_column_logs) == 1 and "churn" in unknown_column_logs[0].getMessage()


def test_exit_1_when_input_file_is_missing(tmp_path):
    assert run(tmp_path / "nope.csv", tmp_path / "out.csv") == 1


def test_exit_1_when_input_file_is_empty(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert run(empty, tmp_path / "out.csv") == 1


def test_exit_1_when_id_column_is_missing(tmp_path):
    no_id = tmp_path / "no_id.csv"
    pd.read_csv(SAMPLE_INPUT).drop(columns="customer_id").to_csv(no_id, index=False)
    assert run(no_id, tmp_path / "out.csv") == 1
    assert not (tmp_path / "out.csv").exists()


def test_exit_3_when_model_cannot_be_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "missing.joblib"))
    assert run(SAMPLE_INPUT, tmp_path / "out.csv") == 3


def test_exit_2_on_bad_command_line():
    with pytest.raises(SystemExit) as info:
        main(["--input", str(SAMPLE_INPUT)])  # --output missing
    assert info.value.code == 2
