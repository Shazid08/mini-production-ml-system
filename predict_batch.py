"""Batch scoring CLI: score a CSV of unseen customers.

Usage (from the repository root):
    python predict_batch.py --input data/sample_input.csv --output predictions.csv

Input: a CSV with a `customer_id` column and the 19 feature columns named as in
src/data_pipeline.FEATURES, in any order. Other columns are ignored (one warning).
Blank or whitespace-only cells count as missing.

Rows are validated with the same schema rules as the API (serving/schemas.py),
then prepared and scored with the same code as the API (serving/scoring.py).
Differences from the API, by design: CSV text is parsed into numbers, and
unseen category values are scored with a row-level warning instead of rejected.
A row that fails validation does not stop the run; it is written to the output
with status "rejected", a reason, and no score.

Output CSV columns:
  row_number, customer_id, status (scored / rejected), reason, churn_probability,
  churn_prediction, decision_rule, source, fallback_reason, warnings

Exit codes: 0 = finished (even if some rows were rejected), 1 = input file missing,
unreadable or without required columns, or output not writable, 2 = bad command-line
arguments, 3 = model artifact could not be loaded.
Environment: CHURN_MODEL_PATH overrides the artifact path (default artifacts/churn_model.joblib).
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from serving.schemas import BatchRecord
from serving.scoring import ModelLoadError, load_model, prepare_features, score_frame
from src.data_pipeline import FEATURES, ID_COLUMN, REPO_ROOT

DEFAULT_MODEL_PATH = REPO_ROOT / "artifacts" / "churn_model.joblib"
EXPECTED_COLUMNS = [ID_COLUMN] + FEATURES
OUTPUT_COLUMNS = ["row_number", "customer_id", "status", "reason", "churn_probability", "churn_prediction",
                  "decision_rule", "source", "fallback_reason", "warnings"]
EXIT_INPUT_ERROR, EXIT_MODEL_ERROR = 1, 3

logger = logging.getLogger("predict_batch")


def read_input(path: Path) -> pd.DataFrame:
    """Read every cell as text so validation, not pandas, decides what is a valid number."""
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def validate_rows(df: pd.DataFrame):
    """Validate each row with the shared BatchRecord schema.

    Returns (valid records, their row positions, {position: rejection reason}).
    """
    valid, positions, rejections = [], [], {}
    for pos, row in enumerate(df.to_dict(orient="records")):
        cells = {k: v.strip() for k, v in row.items() if v.strip()}  # blank cell = missing field
        try:
            record = BatchRecord.model_validate(cells)
        except ValidationError as exc:
            rejections[pos] = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or 'record'}: {e['msg']}" for e in exc.errors())
            continue
        valid.append(record.model_dump())
        positions.append(pos)
    return valid, positions, rejections


def run(input_path: Path, output_path: Path) -> int:
    start = time.perf_counter()

    model_path = Path(os.environ.get("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH))
    try:
        bundle = load_model(model_path)
    except ModelLoadError as exc:
        logger.error("Cannot load model from %s: %s", model_path, exc)
        return EXIT_MODEL_ERROR
    logger.info("Model loaded from %s (threshold %.4f)", model_path, bundle["threshold"])

    try:
        df = read_input(input_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.error("Cannot read input file %s: %s: %s", input_path, type(exc).__name__, exc)
        return EXIT_INPUT_ERROR

    unknown = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if unknown:
        logger.warning("Ignoring %d unknown column(s): %s", len(unknown), unknown)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.error("Input file %s is missing required column(s): %s", input_path, missing)
        return EXIT_INPUT_ERROR
    df = df[EXPECTED_COLUMNS]
    logger.info("Read %d row(s) from %s", len(df), input_path)

    # Validation is per row; scoring below is one vectorised call for all valid rows.
    valid, positions, rejections = validate_rows(df)

    # Every row starts as "rejected" with no score; valid rows are filled in below.
    # Each column gets an explicit type up front (pandas will not upcast on assignment).
    n = len(df)
    empty_text = lambda: pd.Series([None] * n, dtype=object)  # noqa: E731
    out = pd.DataFrame({
        "row_number": range(1, n + 1),                 # 1-based data row, for rows with no usable ID
        "customer_id": df[ID_COLUMN].str.strip().to_numpy(),
        "status": pd.Series(["rejected"] * n, dtype=object),
        "reason": pd.Series([rejections.get(i) for i in range(n)], dtype=object),
        "churn_probability": pd.Series([float("nan")] * n, dtype=float),
        "churn_prediction": empty_text(),
        "decision_rule": empty_text(),
        "source": empty_text(),
        "fallback_reason": empty_text(),
        "warnings": empty_text(),
    }, columns=OUTPUT_COLUMNS)

    warned = 0
    if valid:
        frame, row_warnings = prepare_features(valid, bundle)
        result = score_frame(frame, bundle)
        if result.fallback_reason:
            logger.error("FALLBACK (%s) for all %d valid row(s): %s", result.source, len(valid), result.fallback_reason)
        out.loc[positions, "status"] = "scored"
        out.loc[positions, "churn_probability"] = result.probability
        out.loc[positions, "churn_prediction"] = result.prediction
        out.loc[positions, "decision_rule"] = result.decision_rule
        out.loc[positions, "source"] = result.source
        out.loc[positions, "fallback_reason"] = result.fallback_reason
        out.loc[positions, "warnings"] = ["; ".join(w) if w else None for w in row_warnings]
        warned = sum(1 for w in row_warnings if w)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
    except OSError as exc:
        logger.error("Cannot write output file %s: %s", output_path, exc)
        return EXIT_INPUT_ERROR

    elapsed = time.perf_counter() - start
    logger.info("Rows read: %d | scored: %d | rejected: %d | scored with warnings: %d",
                len(df), len(valid), len(rejections), warned)
    logger.info("Elapsed: %.3f s | throughput: %.1f rows/s | output: %s",
                elapsed, len(df) / elapsed if elapsed > 0 else float("nan"), output_path)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Score a CSV of customers with the churn model.")
    parser.add_argument("--input", required=True, type=Path, help="CSV file of customers to score")
    parser.add_argument("--output", required=True, type=Path, help="CSV file to write predictions to")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return run(args.input, args.output)


if __name__ == "__main__":
    sys.exit(main())
