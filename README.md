# Mini Production ML System: Customer Churn Prediction

An end-to-end churn prediction system on the IBM Telco Customer Churn dataset:
leak-free data pipeline, baselines and a class-weighted LightGBM model, a FastAPI
service (`/health`, `/predict`) with a heuristic and prior-rate fallback, a batch
scoring CLI, tests, a latency/throughput benchmark and a Docker image.

## Quickstart

The trained model is committed (`artifacts/churn_model.joblib`), so the API, the batch
CLI and the tests run straight after cloning, with no dataset download.

```bash
git clone https://github.com/Shazid08/mini-production-ml-system.git
cd mini-production-ml-system
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m pytest                   # 111 passed, 3 skipped (the 3 need the Kaggle data)
python predict_batch.py --input data/sample_input.csv --output data/output/predictions.csv
uvicorn serving.app:app --port 8000  # then try the curl examples in section 4
```

To reproduce the data split and the model from scratch, follow sections 2 and 3.

## 1. Setup

Requires Python 3.12 (developed on 3.12.7).

```bash
git clone https://github.com/Shazid08/mini-production-ml-system.git
cd mini-production-ml-system
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins every package (direct and transitive) to an exact version.

## 2. Get the data

The raw dataset is not stored in this repository. It is the IBM Telco Customer
Churn dataset, published on Kaggle as
[`yeanzc/telco-customer-churn-ibm-dataset`](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).
Download it with:

```bash
python -m src.data_pipeline
```

This needs internet access. It uses `kagglehub` to place `Telco_customer_churn.xlsx` (7,043 rows, 33 columns)
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
Training is seeded, so re-running it gives the same model predictions and metrics.

## 4. Online API (FastAPI)

```bash
uvicorn serving.app:app --port 8000
```

The model artifact is loaded once at startup. Endpoints:

- `GET /health`: `200` with the model threshold when the model is loaded, `503` otherwise.
- `POST /predict`: scores one customer.

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"threshold":0.6802878662683045,"fallback_probability":0.2653144016227181,"simulated_failure":null,"detail":null}
```

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

```json
{"request_id":"req-001","customer_id":"CUST-1001","churn_probability":0.8746326289659453,"churn_prediction":true,
 "threshold":0.6802878662683045,"decision_rule":"model probability >= 0.6803","source":"model","fallback_reason":null,"warnings":[]}
```

A malformed request is rejected with `422` and a list of every problem, for example
sending `"tenure_months": "12"` (a string) and `"contract": "Three year"` (not an allowed value)
(response abbreviated):

```json
{"detail":[{"type":"int_type","loc":["body","tenure_months"],"msg":"Input should be a valid integer","input":"12"},
 {"type":"literal_error","loc":["body","contract"],"msg":"Input should be 'Month-to-month', 'One year' or 'Two year'", ...}]}
```

Interactive API documentation is served at `http://127.0.0.1:8000/docs`.

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
until `python -m src.data_pipeline` has been run, so a fresh clone shows
`111 passed, 3 skipped`.

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

Results from one run (2026-11-12): Apple M3 Pro, 18 GB RAM, macOS 27.0.1, Python 3.12.7, AC Power, client and server on localhost, one uvicorn worker. Values are medians of 3 repeats, rounded.

| API, 1,000 requests per run | p50 ms | p95 ms | p99 ms | mean ms | min ms | max ms | requests/s | errors |
|---|---|---|---|---|---|---|---|---|
| 1 client (sequential) | 4.05 | 4.22 | 4.33 | 4.10 | 3.85 | 23.75 | 243.6 | 0 |
| 8 concurrent clients | 29.68 | 59.29 | 76.70 | 31.65 | 4.12 | 116.63 | 248.3 | 0 |

- Server start-up to healthy: 1.22 s; first (cold) request: 44.38 ms (20 warm-up requests, excluded above).
- Model time without HTTP, one record: full request path p50 3.30 ms, of which `predict_proba` p50 2.00 ms.
- Batch, 100,000 rows (repeated dataset rows): whole run including start-up 2.50 s (40,036 rows/s); scoring only 82,065 rows/s, split into validation 0.94 s and prediction 0.17 s.

## 8. Docker (optional)

Everything above also works without Docker. The image runs the API by default and
contains the batch CLI. It is built on `python:3.12.7-slim-bookworm`
(digest `sha256:60d9996b6a8a3689d36db740b49f4327be3be09a21122bd02fb8895abb38b50d`),
installs the pinned `requirements.txt` plus `libgomp1` (the OpenMP runtime that
LightGBM needs on slim images), and runs as a non-root user.

```bash
docker build -t churn-api .
docker run -d --name churn-api -p 8000:8000 churn-api
curl -s http://127.0.0.1:8000/health
```

`docker ps` shows the container as `healthy` once the built-in `HEALTHCHECK`
(a request to `/health` every 30 s) succeeds. `/predict` takes the same request as in
section 4.

Batch scoring with a mounted folder (input and output files live on the host):

```bash
mkdir -p data/io && cp data/sample_input.csv data/io/
docker run --rm -v "$PWD/data/io:/io" churn-api \
  python predict_batch.py --input /io/sample_input.csv --output /io/predictions.csv
```

On a Linux host, the mounted folder keeps your host file ownership, so the image's
non-root user (uid 10001) may be unable to write to it, and any files it does write are
owned by uid 10001. Run the container as your own user instead:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/data/io:/io" churn-api \
  python predict_batch.py --input /io/sample_input.csv --output /io/predictions.csv
```

(Docker Desktop on macOS and Windows maps file ownership itself, so this is not needed there.)

The benchmark numbers in section 7 were measured natively, not in Docker: Docker on
macOS runs containers inside a Linux virtual machine, which adds its own overhead.

## Repository layout

```
mini-production-ml-system/
├── README.md
├── requirements.txt         pinned dependencies
├── Dockerfile, .dockerignore
├── pytest.ini
├── predict_batch.py         batch scoring CLI
├── data/
│   └── sample_input.csv     20 test-split customers (the full dataset is downloaded, not committed)
├── artifacts/
│   ├── churn_model.joblib   fitted pipeline + threshold + fallback settings
│   ├── metrics.json         grid search, validation/test metrics, cost analysis, bootstrap CI
│   ├── benchmark_results.json
│   └── figures/             test-set PR curve and confusion matrix
├── src/
│   ├── data_pipeline.py     download, leakage audit, cleaning, stratified split
│   ├── train.py             baselines, LightGBM search, threshold choice, test evaluation, export
│   └── evaluate.py          metrics, threshold rule, cost view, bootstrap, figures
├── serving/
│   ├── app.py               FastAPI service (/health, /predict)
│   ├── schemas.py           Pydantic request/response schemas and validation rules
│   ├── scoring.py           shared model loading, feature preparation, scoring with fallback
│   └── benchmark.py         latency and throughput benchmark
└── tests/                   pytest suite and fixtures
```
