"""Shared test payloads: one valid API request and the list of payloads that must be rejected."""

import math

VALID_PAYLOAD = {
    "request_id": "req-test", "customer_id": "CUST-TEST",
    "tenure_months": 4, "monthly_charges": 89.65, "total_charges": 358.6,
    "gender": "Female", "senior_citizen": "No", "partner": "No", "dependents": "No",
    "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
    "online_security": "No", "online_backup": "No", "device_protection": "No",
    "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "No",
    "contract": "Month-to-month", "paperless_billing": "Yes", "payment_method": "Electronic check",
}

DROP = object()  # marker: remove this key from the payload


def with_changes(changes: dict) -> dict:
    payload = dict(VALID_PAYLOAD)
    for key, value in changes.items():
        if value is DROP:
            payload.pop(key)
        else:
            payload[key] = value
    return payload


# (case id, changes to the valid payload, field the error must point at; "" = whole record)
REJECTED_CASES = [
    ("missing required field", {"contract": DROP}, "contract"),
    ("missing customer_id", {"customer_id": DROP}, "customer_id"),
    ("empty customer_id", {"customer_id": ""}, "customer_id"),
    ("tenure as a string", {"tenure_months": "12"}, "tenure_months"),
    ("fractional tenure", {"tenure_months": 12.5}, "tenure_months"),
    ("negative monthly charge", {"monthly_charges": -5}, "monthly_charges"),
    ("tenure above 1200 cap", {"tenure_months": 5000}, "tenure_months"),
    ("monthly charge above cap", {"monthly_charges": 20000.0}, "monthly_charges"),
    ("negative total charges", {"total_charges": -1.0}, "total_charges"),
    ("unseen category", {"contract": "Three year"}, "contract"),
    ("wrong-case category", {"contract": "month-to-month"}, "contract"),
    ("null categorical", {"gender": None}, "gender"),
    ("null total with tenure > 0", {"tenure_months": 5, "total_charges": None}, ""),
    ("unknown extra field", {"churn_score": 90}, "churn_score"),
]

# Not expressible in standard JSON, so checked at the schema level only.
SCHEMA_ONLY_REJECTED_CASES = [
    ("NaN monthly charge", {"monthly_charges": math.nan}, "monthly_charges"),
    ("infinite total charges", {"total_charges": math.inf}, "total_charges"),
]
