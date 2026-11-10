# Mini Production ML System: Customer Churn Prediction

> Work in progress. Training, serving, batch scoring, testing and Docker
> instructions will be added as those parts are built.

## 1. Setup

Requires Python 3.12 (developed on 3.12.7).

```bash
git clone <repo-url> mini-production-ml-system
cd mini-production-ml-system
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the data

The raw dataset is not stored in this repository. It is the IBM Telco Customer
Churn dataset, published on Kaggle as
[`yeanzc/telco-customer-churn-ibm-dataset`](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).
Download it with:

```bash
python -m src.data_pipeline
```

This uses `kagglehub` to place `Telco_customer_churn.xlsx` (7,043 rows, 33 columns)
in `data/raw/`. It is a public dataset, so no Kaggle account or API key is needed.
The same command then prints the leakage-audit evidence, cleans the data and writes a
stratified 70/15/15 split (seed 42) to `data/processed/{train,val,test}.csv`.

## 3. Train and evaluate

```bash
python -m src.train
```

Fits the baselines (dummy, heuristic rule, logistic regression) and a LightGBM
pipeline on the training split, runs a 24-candidate LightGBM grid search scored by
validation PR-AUC, chooses each model's decision threshold on validation, then
evaluates every model once on the test split. Outputs:

- `artifacts/churn_model.joblib`: fitted preprocessing + LightGBM pipeline and its threshold
- `artifacts/metrics.json`: search results, validation and test metrics, cost analysis
- `artifacts/figures/`: test-set PR curve and confusion matrix

`python -m src.train --skip-test` runs everything except the test evaluation.

## 4. Online API (FastAPI)

```bash
uvicorn serving.app:app --port 8000
```

The model artifact is loaded once at startup. Endpoints:

- `GET /health`: `200` with the model threshold when the model is loaded, `503` otherwise.
- `POST /predict`: scores one customer.

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "request_id": "req-001", "customer_id": "CUST-1001",
  "tenure_months": 4, "monthly_charges": 89.65, "total_charges": 358.6,
  "gender": "Female", "senior_citizen": "No", "partner": "No", "dependents": "No",
  "phone_service": "Yes", "multiple_lines": "No", "internet_service": "Fiber optic",
  "online_security": "No", "online_backup": "No", "device_protection": "No",
  "tech_support": "No", "streaming_tv": "Yes", "streaming_movies": "No",
  "contract": "Month-to-month", "paperless_billing": "Yes", "payment_method": "Electronic check"}'
```

Validation rules (violations return `422`): every field is required except
`total_charges`, which may be `null` only when `tenure_months` is 0; categorical
values must be one of the values seen in the data (listed in
`src/data_pipeline.py: ALLOWED_CATEGORIES`); negative numbers, NaN/infinity and
impossible values (tenure above 1,200 months, monthly charges above 10,000) are
rejected; unknown fields are rejected. Values outside the training range are
accepted, logged and returned in `warnings`.

Fallback (the response is still `200`; `source`, `decision_rule` and `fallback_reason` say what happened):

1. `model`: LightGBM probability, churn if `>= threshold` (0.6803).
2. `fallback_heuristic`: used if the model raises. Churn if the rule score is 2
   (month-to-month contract and tenure <= 12 months); `churn_probability` is the
   training churn rate of customers with the same rule score.
3. `fallback_prior_rate`: used if the heuristic also raises. `churn_probability`
   is the training churn rate (0.2653).

`CHURN_MODEL_PATH` loads a different artifact. `CHURN_API_SIMULATE_FAILURE` is a
**test-only** fault-injection switch, unset (off) by default: `model` forces layer 2,
`all` forces layer 3. `/health` shows its state in `simulated_failure`.

## 5. Batch scoring CLI

```bash
python predict_batch.py --input data/sample_input.csv --output data/output/predictions.csv
```

`data/sample_input.csv` holds 20 customers from the test split (no target column).
The input needs a `customer_id` column and the 19 feature columns (any order);
other columns are ignored with a warning. Rows use the same validation rules and
scoring code as the API, with two deliberate differences: numbers are parsed from
CSV text, and unseen category values are scored with a row-level warning instead
of being rejected. Invalid rows do not stop the run: they appear in the output with
`status=rejected`, a `reason` and no score. Output columns: `row_number`,
`customer_id`, `status`, `reason`, `churn_probability`, `churn_prediction`,
`decision_rule`, `source`, `fallback_reason`, `warnings`.

Exit codes: `0` finished, `1` input missing/unreadable/missing columns or output not
writable, `2` bad arguments, `3` model artifact could not be loaded.

## 6. Run the tests

```bash
python -m pytest
```

114 tests cover the schemas (every valid category value, every rejected payload),
the API (status codes, IDs, warnings, all three fallback layers, `503` without an
artifact), inference integrity (the artifact reproduces its stored reference
predictions, probabilities in [0, 1], no leak columns among the features, identical
API and batch output), the batch CLI (messy-file handling and exit codes) and the
split logic. Three tests need the downloaded dataset and are skipped, with a message,
until `python -m src.data_pipeline` has been run.

## 7. Benchmark

```bash
python -m serving.benchmark
```

Starts the API on localhost and sends 20 untimed warm-up requests (reported as cold
start), then 1,000 timed requests from one client and 1,000 from 8 concurrent
clients, each run repeated 3 times. It also times the model without HTTP and runs
the batch CLI on a 100,000-row file built by repeating dataset rows (a throughput
test, not an accuracy test). Raw results and the machine/library details are saved
to `artifacts/benchmark_results.json`. The numbers depend on the machine they run on.
