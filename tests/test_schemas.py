"""Schema rules (serving/schemas.py) for the API and the batch variant."""

import pytest
from pydantic import ValidationError

from payloads import DROP, REJECTED_CASES, SCHEMA_ONLY_REJECTED_CASES, VALID_PAYLOAD, with_changes
from serving.schemas import BatchRecord, PredictRequest
from src.data_pipeline import ALLOWED_CATEGORIES


def test_valid_payload_is_accepted():
    request = PredictRequest.model_validate(VALID_PAYLOAD)
    assert request.customer_id == "CUST-TEST"


@pytest.mark.parametrize("feature,value", [(f, v) for f, values in ALLOWED_CATEGORIES.items() for v in values])
def test_every_allowed_category_value_is_accepted(feature, value):
    # Includes "No internet service" and "No phone service".
    PredictRequest.model_validate(with_changes({feature: value}))


@pytest.mark.parametrize("total_charges", [None, DROP], ids=["null", "omitted"])
def test_total_charges_may_be_missing_for_tenure_zero(total_charges):
    payload = with_changes({"tenure_months": 0, "total_charges": total_charges})
    assert PredictRequest.model_validate(payload).total_charges is None


@pytest.mark.parametrize("case,changes,field", REJECTED_CASES + SCHEMA_ONLY_REJECTED_CASES,
                         ids=[c[0] for c in REJECTED_CASES + SCHEMA_ONLY_REJECTED_CASES])
def test_invalid_payloads_are_rejected(case, changes, field):
    with pytest.raises(ValidationError) as info:
        PredictRequest.model_validate(with_changes(changes))
    locations = [".".join(str(p) for p in e["loc"]) for e in info.value.errors()]
    assert field in locations


def _batch_row(**changes):
    row = {k: str(v) for k, v in VALID_PAYLOAD.items() if k != "request_id"}
    row.update(changes)
    return row


def test_batch_schema_parses_numbers_from_text():
    record = BatchRecord.model_validate(_batch_row(tenure_months="12", monthly_charges="70.5"))
    assert record.tenure_months == 12 and record.monthly_charges == 70.5


def test_batch_schema_accepts_unseen_category_for_lenient_scoring():
    assert BatchRecord.model_validate(_batch_row(contract="Three year")).contract == "Three year"


@pytest.mark.parametrize("changes", [{"tenure_months": "12.5"}, {"tenure_months": "abc"},
                                     {"monthly_charges": "-20"}, {"monthly_charges": "nan"}, {"contract": ""}])
def test_batch_schema_keeps_the_shared_rules(changes):
    with pytest.raises(ValidationError):
        BatchRecord.model_validate(_batch_row(**changes))
