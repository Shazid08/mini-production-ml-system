"""Data pipeline: ingestion, cleaning, leakage audit and leak-free splitting.

Run from the repository root:
    python -m src.data_pipeline

Steps:
  1. Download the raw Excel file from Kaggle with kagglehub (skipped if present).
  2. Split the RAW rows into stratified train/validation/test (70/15/15, fixed seed).
  3. Print the leakage-audit evidence, computed on the TRAINING partition only.
  4. Clean each partition (drop audited columns, fix Total Charges, rename columns).
  5. Save data/processed/{train,val,test}.csv.

Why split before cleaning: cleaning here is stateless and row-by-row (no means,
vocabularies or other statistics are learned), so it gives the same result either
way. Splitting first means the audit, which looks at column-vs-target
relationships, never sees validation or test labels.

The raw dataset is NOT stored in Git; step 1 recreates it for anyone who clones
the repository.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Kaggle handle of the IBM Telco Customer Churn dataset (public dataset page:
# https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset).
KAGGLE_HANDLE = "yeanzc/telco-customer-churn-ibm-dataset"
RAW_FILENAME = "Telco_customer_churn.xlsx"  # verified after the first download

# Paths are built relative to the repository root, so the script works no matter
# which directory it is launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

RANDOM_SEED = 42
RAW_TARGET = "Churn Value"  # 1 = churned, 0 = stayed

# Leakage / relevance audit. Every raw column that is NOT used as a model input
# is listed here with the reason. The evidence behind each reason is printed by
# print_audit_evidence() and reproduced in the report's audit table.
DROPPED_COLUMNS = {
    "CustomerID": "Identifier. Kept as the record ID in outputs, never used as a feature.",
    "Churn Label": "Target in text form (Yes/No); maps 1-to-1 onto Churn Value.",
    "Churn Score": "IBM-derived score; separates the classes almost perfectly (not available for a new customer).",
    "CLTV": "IBM-derived value from an undocumented formula; cannot confirm it is free of outcome information.",
    "Churn Reason": "Recorded only for customers who already churned; its presence equals the label.",
    "Count": "Constant (always 1).",
    "Country": "Constant (always 'United States').",
    "State": "Constant (always 'California').",
    "City": "1,129 categories, 86% with 5 or fewer customers; too sparse to learn from.",
    "Zip Code": "1,652 categories with at most 5 customers each; behaves like an identifier.",
    "Lat Long": "Text copy of Latitude + Longitude (one-to-one with Zip Code).",
    "Latitude": "Location of the zip code; no univariate signal and too few customers per location.",
    "Longitude": "Location of the zip code; no univariate signal and too few customers per location.",
}

# Kept columns, renamed to snake_case so they are convenient as API field names.
RENAME_MAP = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Senior Citizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "Tenure Months": "tenure_months",
    "Phone Service": "phone_service",
    "Multiple Lines": "multiple_lines",
    "Internet Service": "internet_service",
    "Online Security": "online_security",
    "Online Backup": "online_backup",
    "Device Protection": "device_protection",
    "Tech Support": "tech_support",
    "Streaming TV": "streaming_tv",
    "Streaming Movies": "streaming_movies",
    "Contract": "contract",
    "Paperless Billing": "paperless_billing",
    "Payment Method": "payment_method",
    "Monthly Charges": "monthly_charges",
    "Total Charges": "total_charges",
    "Churn Value": "churn",
}

ID_COLUMN = "customer_id"
TARGET = "churn"
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [
    "gender", "senior_citizen", "partner", "dependents", "phone_service",
    "multiple_lines", "internet_service", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "contract", "paperless_billing", "payment_method",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Every permitted value of each categorical feature (checked against the raw file).
# Single source of truth: cleaning rejects any other value, the API schema accepts
# exactly these values, and the API checks them against the fitted encoder at startup.
_NO_YES = ["No", "Yes"]
_INTERNET_ADDON = ["No", "Yes", "No internet service"]
ALLOWED_CATEGORIES = {
    "gender": ["Female", "Male"],
    "senior_citizen": _NO_YES,
    "partner": _NO_YES,
    "dependents": _NO_YES,
    "phone_service": _NO_YES,
    "multiple_lines": ["No", "Yes", "No phone service"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": _INTERNET_ADDON,
    "online_backup": _INTERNET_ADDON,
    "device_protection": _INTERNET_ADDON,
    "tech_support": _INTERNET_ADDON,
    "streaming_tv": _INTERNET_ADDON,
    "streaming_movies": _INTERNET_ADDON,
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": _NO_YES,
    "payment_method": ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
}
assert list(ALLOWED_CATEGORIES) == CATEGORICAL_FEATURES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- ingestion
def download_raw_data(output_dir: Path = RAW_DIR) -> list[Path]:
    """Download the dataset files into `output_dir` and return their paths.

    kagglehub skips the download if the files are already present in
    `output_dir`, so calling this repeatedly is cheap.
    """
    import kagglehub  # imported here so the serving code does not need it at runtime

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s into %s", KAGGLE_HANDLE, output_dir)
    kagglehub.dataset_download(KAGGLE_HANDLE, output_dir=str(output_dir))

    # Report what actually arrived instead of assuming a file name. kagglehub also
    # writes a hidden ".complete" marker folder; skip hidden paths in the listing.
    files = sorted(
        p for p in output_dir.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(output_dir).parts)
    )
    for path in files:
        logger.info("  %s (%d bytes)", path.relative_to(REPO_ROOT), path.stat().st_size)
    return files


def load_raw(path: Path = RAW_DIR / RAW_FILENAME) -> pd.DataFrame:
    """Read the raw workbook and check that every expected column is present."""
    df = pd.read_excel(path)
    expected = set(DROPPED_COLUMNS) | set(RENAME_MAP)
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {sorted(missing)}")
    logger.info("Loaded %s: %d rows x %d columns", path.name, *df.shape)
    return df


# ---------------------------------------------------------------- splitting
def split_data(df: pd.DataFrame, target: str = RAW_TARGET, seed: int = RANDOM_SEED):
    """Stratified 70/15/15 split into train, validation and test.

    The data has no timestamp, so a temporal cut-off is impossible. Stratifying
    on the target keeps the churn rate (about 26.5%) the same in every partition.
    Done in two stages: 70% train vs 30% holdout, then the holdout in half.
    """
    train, holdout = train_test_split(df, test_size=0.30, stratify=df[target], random_state=seed)
    val, test = train_test_split(holdout, test_size=0.50, stratify=holdout[target], random_state=seed)
    return train, val, test


# ---------------------------------------------------------------- leakage audit
def print_audit_evidence(train_raw: pd.DataFrame, full_raw: pd.DataFrame) -> None:
    """Log the numbers behind each decision in DROPPED_COLUMNS.

    Anything that involves the target uses the training partition only.
    Pure column-to-column mapping checks (no target) use the full file.
    """
    y = train_raw[RAW_TARGET]
    log = logger.info
    log("=== Leakage audit evidence (target checks on TRAIN, n=%d) ===", len(train_raw))

    # Two forms of the target.
    log("Churn Label vs Churn Value:\n%s", pd.crosstab(train_raw["Churn Label"], y).to_string())

    # Churn Score.
    by_class = train_raw.groupby(y)["Churn Score"].agg(["min", "max", "mean"]).round(1)
    log("Churn Score by class (0=stay, 1=churn):\n%s", by_class.to_string())
    log("Churn Score ROC-AUC on its own: %.4f", roc_auc_score(y, train_raw["Churn Score"]))
    churn_min = train_raw.loc[y == 1, "Churn Score"].min()
    log("Every churner has Churn Score >= %d; rule 'score >= %d' recall = %.3f",
        churn_min, churn_min, (train_raw.loc[y == 1, "Churn Score"] >= churn_min).mean())

    # CLTV.
    log("CLTV mean by class: stay %.1f, churn %.1f; ROC-AUC on its own: %.4f",
        train_raw.loc[y == 0, "CLTV"].mean(), train_raw.loc[y == 1, "CLTV"].mean(),
        roc_auc_score(y, train_raw["CLTV"]))
    log("corr(CLTV, Tenure Months) = %.3f", train_raw["CLTV"].corr(train_raw["Tenure Months"]))

    # Churn Reason.
    present = train_raw["Churn Reason"].notna().rename("Churn Reason present")
    log("Churn Reason presence vs target:\n%s", pd.crosstab(present, y).to_string())

    # Geography: mapping checks (no target) on the full file.
    for a, b in [("Zip Code", "Lat Long"), ("Lat Long", "Zip Code"), ("Zip Code", "City")]:
        log("Max distinct %s per %s: %d", b, a, full_raw.groupby(a)[b].nunique().max())
    log("Max distinct Zip Code per City: %d", full_raw.groupby("City")["Zip Code"].nunique().max())
    per_zip = full_raw["Zip Code"].value_counts()
    per_city = full_raw["City"].value_counts()
    log("Zip Code: %d categories, max %d customers each", per_zip.size, per_zip.max())
    log("City: %d categories, %.1f%% with <= 5 customers", per_city.size, 100 * (per_city <= 5).mean())
    log("Latitude ROC-AUC on its own: %.4f; Longitude: %.4f",
        roc_auc_score(y, train_raw["Latitude"]), roc_auc_score(y, train_raw["Longitude"]))

    # Constant columns.
    for col in ["Count", "Country", "State"]:
        log("%s unique values: %s", col, full_raw[col].unique().tolist())


# ---------------------------------------------------------------- cleaning
def fill_new_customer_total_charges(total_charges: pd.Series, tenure_months: pd.Series) -> pd.Series:
    """Missing total charges for a tenure-0 customer means "not billed yet", so 0.0.
    Shared by training-data cleaning and the API so both apply the same rule."""
    return total_charges.mask(total_charges.isna() & (tenure_months == 0), 0.0)


def check_categories(df: pd.DataFrame) -> None:
    """Raise if any categorical column holds a value outside ALLOWED_CATEGORIES."""
    for feature, allowed in ALLOWED_CATEGORIES.items():
        unexpected = set(df[feature].dropna().unique()) - set(allowed)
        if unexpected:
            raise ValueError(f"{feature}: values not in ALLOWED_CATEGORIES: {sorted(unexpected)}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, row-wise cleaning. Learns nothing from the data.

    - Drops the audited columns (CustomerID is kept as the record ID).
    - Total Charges: the raw file stores 11 blank strings (" "), all for
      customers with tenure 0 who have not been billed yet. Blanks become NaN,
      and NaN with tenure 0 becomes 0.0. Any other NaN is left for the
      model pipeline's imputer, which is fit on the training data only.
    - Renames columns to snake_case.
    """
    out = df.drop(columns=[c for c in DROPPED_COLUMNS if c != "CustomerID"])
    total = pd.to_numeric(out["Total Charges"], errors="coerce")  # " " -> NaN
    out["Total Charges"] = fill_new_customer_total_charges(total, out["Tenure Months"]).astype(float)
    out = out.rename(columns=RENAME_MAP)
    check_categories(out)
    return out[[ID_COLUMN] + FEATURES + [TARGET]]


# ---------------------------------------------------------------- main
def summarise(name: str, part: pd.DataFrame, total_rows: int) -> None:
    churners = int(part[TARGET].sum())
    logger.info("%-5s rows=%5d (%.1f%%)  churn=1: %4d  churn=0: %4d  churn rate=%.2f%%",
                name, len(part), 100 * len(part) / total_rows,
                churners, len(part) - churners, 100 * part[TARGET].mean())


def main() -> None:
    download_raw_data()
    raw = load_raw()

    train_raw, val_raw, test_raw = split_data(raw)
    print_audit_evidence(train_raw, raw)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== Cleaned partitions (seed=%d) ===", RANDOM_SEED)
    for name, part in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
        cleaned = clean(part)
        cleaned.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        summarise(name, cleaned, len(raw))

    # Sanity checks: no customer in two partitions, and no rows lost.
    ids = [set(clean(p)[ID_COLUMN]) for p in (train_raw, val_raw, test_raw)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "partitions overlap"
    assert sum(len(i) for i in ids) == len(raw), "rows lost during split"
    logger.info("Checks passed: partitions are disjoint and cover all %d rows", len(raw))
    logger.info("Saved to %s", PROCESSED_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
