"""Model training, threshold selection, final test evaluation and export.

Run from the repository root (after python -m src.data_pipeline):
    python -m src.train              # full run, including the single test evaluation
    python -m src.train --skip-test  # development run: test set is never loaded

What happens, in order:
  1. Fit three baselines (dummy, heuristic rule, logistic regression) on TRAIN.
  2. Small LightGBM grid search: each candidate fit on TRAIN, scored by PR-AUC on VALIDATION.
  3. Choose every model's threshold on VALIDATION with the same rule (evaluate.choose_threshold).
  4. Cost view on VALIDATION with clearly assumed costs, plus a sensitivity grid.
  5. Feature importances of the final LightGBM model.
  6. (Full run only) Evaluate all four models ONCE on TEST at their frozen
     thresholds; save the confusion matrix and PR curve figures.
  7. Save the LightGBM pipeline + threshold with joblib to artifacts/churn_model.joblib.
"""

import argparse
import itertools
import json
import logging
import platform

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_pipeline import CATEGORICAL_FEATURES, FEATURES, ID_COLUMN, NUMERIC_FEATURES, PROCESSED_DIR, REPO_ROOT, TARGET
from src.evaluate import (ASSUMED_CONTACT_COST, ASSUMED_SAVED_VALUE, TARGET_PRECISION, best_cost_threshold,
                      choose_threshold, evaluate, net_benefit, paired_bootstrap_pr_auc, plot_confusion_matrix,
                      plot_pr_curves)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
MODEL_PATH = ARTIFACTS_DIR / "churn_model.joblib"
RANDOM_SEED = 42

# Small, documented LightGBM search space (3 x 2 x 2 x 2 = 24 candidates).
LGBM_GRID = {
    "num_leaves": [7, 15, 31],          # tree complexity
    "learning_rate": [0.03, 0.1],       # step size per tree
    "n_estimators": [100, 300],         # number of trees
    "min_child_samples": [20, 50],      # minimum customers per leaf (limits overfitting)
}

# Sensitivity grid for the cost view (all values are assumptions).
CONTACT_COSTS = [25.0, 50.0, 100.0]
SAVED_VALUES = [150.0, 300.0, 600.0]

MODEL_NAMES = ["dummy_prior", "heuristic_rule", "logistic_regression", "lightgbm"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- data and preprocessing
def load_split(name: str):
    """Return (X, y) for one saved partition ('train', 'val' or 'test')."""
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", dtype={ID_COLUMN: str})
    return df[FEATURES], df[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Imputers, scaler and encoder. Unfitted here; they learn their statistics
    (medians, means, category lists) only when the pipeline is fit on train."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        # Categories never seen in training become all-zero columns instead of an error.
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def heuristic_score(X: pd.DataFrame) -> pd.Series:
    """Rule fixed before seeing results: one point for a month-to-month contract,
    one point for tenure of 12 months or less. Score 0, 1 or 2."""
    month_to_month = (X["contract"] == "Month-to-month").astype(int)
    new_customer = (X["tenure_months"] <= 12).astype(int)
    return month_to_month + new_customer


def make_lgbm_pipeline(params: dict) -> Pipeline:
    """Same preprocessing as logistic regression, LightGBM with class weighting.
    'balanced' weights each class by n_samples / (2 * n_class_samples), so a
    missed churner costs about 2.8x a false alarm in the training loss."""
    model = LGBMClassifier(class_weight="balanced", random_state=RANDOM_SEED, verbose=-1, **params)
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


# ---------------------------------------------------------------- training
def tune_lightgbm(X_train, y_train, X_val, y_val):
    """Fit every grid candidate on train; rank by validation PR-AUC."""
    from src.evaluate import pr_auc
    rows = []
    keys = list(LGBM_GRID)
    for values in itertools.product(*(LGBM_GRID[k] for k in keys)):
        params = dict(zip(keys, values))
        pipe = make_lgbm_pipeline(params).fit(X_train, y_train)
        rows.append({**params, "val_pr_auc": pr_auc(y_val, pipe.predict_proba(X_val)[:, 1])})
    table = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False, kind="stable").reset_index(drop=True)
    best_params = {k: table.loc[0, k] for k in keys}
    best_params = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best_params.items()}
    return best_params, table


def fit_models(X_train, y_train, X_val, y_val):
    """Return ({model name: scoring function}, final LightGBM pipeline, search table)."""
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    logreg = Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]).fit(X_train, y_train)
    best_params, search = tune_lightgbm(X_train, y_train, X_val, y_val)
    lgbm = make_lgbm_pipeline(best_params).fit(X_train, y_train)

    scorers = {
        "dummy_prior": lambda X: dummy.predict_proba(X)[:, 1],
        "heuristic_rule": lambda X: heuristic_score(X).to_numpy(dtype=float),
        "logistic_regression": lambda X: logreg.predict_proba(X)[:, 1],
        "lightgbm": lambda X: lgbm.predict_proba(X)[:, 1],
    }
    return scorers, lgbm, best_params, search


# ---------------------------------------------------------------- analysis
def cost_sensitivity(y_val, val_scores, chosen_threshold):
    """For each assumed (contact cost, saved value) pair, the threshold that
    maximises validation net benefit, compared with the precision-rule threshold."""
    rows = []
    for c, v in itertools.product(CONTACT_COSTS, SAVED_VALUES):
        thr, best = best_cost_threshold(y_val, val_scores, c, v)
        flagged = val_scores >= thr
        tp = int((flagged & (y_val.to_numpy() == 1)).sum())
        rows.append({
            "contact_cost": c, "saved_value": v, "cost_ratio_c_over_v": round(c / v, 3),
            "best_threshold": thr, "contacted": int(flagged.sum()),
            "precision": tp / flagged.sum() if flagged.sum() else float("nan"),
            "recall": tp / int(y_val.sum()),
            "best_net_benefit": best,
            "net_benefit_at_chosen_threshold": net_benefit(y_val, (val_scores >= chosen_threshold).astype(int), c, v),
        })
    return pd.DataFrame(rows)


def feature_importances(lgbm: Pipeline):
    """Gain and split importances per encoded column, and gain summed back to the
    19 original input features (one-hot columns grouped by their source feature)."""
    pre = lgbm.named_steps["preprocess"]
    booster = lgbm.named_steps["model"].booster_
    encoded_names = pre.get_feature_names_out()
    encoder = pre.named_transformers_["cat"].named_steps["onehot"]
    source = list(NUMERIC_FEATURES)
    for feature, cats in zip(CATEGORICAL_FEATURES, encoder.categories_):
        source += [feature] * len(cats)
    encoded = pd.DataFrame({
        "encoded_feature": encoded_names, "source_feature": source,
        "gain": booster.feature_importance("gain"), "splits": booster.feature_importance("split"),
    })
    grouped = encoded.groupby("source_feature")[["gain", "splits"]].sum()
    grouped["gain_share_%"] = 100 * grouped["gain"] / grouped["gain"].sum()
    return encoded.sort_values("gain", ascending=False), grouped.sort_values("gain", ascending=False)


# ---------------------------------------------------------------- main
def main(skip_test: bool) -> None:
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    logger.info("Train: %d rows, Validation: %d rows", len(X_train), len(X_val))

    scorers, lgbm, best_params, search = fit_models(X_train, y_train, X_val, y_val)
    logger.info("LightGBM grid search (24 candidates, fit on train, ranked by validation PR-AUC), top 8:\n%s",
                search.head(8).to_string(float_format=lambda v: f"{v:.4f}"))
    logger.info("Worst candidate validation PR-AUC: %.4f", search["val_pr_auc"].min())
    logger.info("Selected LightGBM params: %s", best_params)

    # Thresholds chosen on validation with one rule, then frozen.
    val_scores = {name: scorers[name](X_val) for name in MODEL_NAMES}
    thresholds, rules, val_metrics = {}, {}, {}
    for name in MODEL_NAMES:
        thr, rule = choose_threshold(y_val, val_scores[name])
        thresholds[name], rules[name] = thr, rule
        val_metrics[name] = evaluate(y_val, val_scores[name], (val_scores[name] >= thr).astype(int))
    val_table = pd.DataFrame(val_metrics).T.astype({"tp": int, "fp": int, "fn": int, "tn": int})
    val_table.insert(0, "threshold", pd.Series(thresholds))
    cols = ["threshold", "pr_auc", "recall_at_precision", "f1_macro", "tp", "fp", "fn", "tn"]
    logger.info("VALIDATION metrics at validation-chosen thresholds:\n%s",
                val_table[cols].to_string(float_format=lambda v: f"{v:.4f}"))
    for name in MODEL_NAMES:
        logger.info("  threshold rule used for %-20s %s", name + ":", rules[name])

    # Cost view (assumed costs).
    costs = cost_sensitivity(y_val, val_scores["lightgbm"], thresholds["lightgbm"])
    logger.info("Cost view on VALIDATION, LightGBM. ASSUMED costs; default c=%.0f, V=%.0f:\n%s",
                ASSUMED_CONTACT_COST, ASSUMED_SAVED_VALUE, costs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Feature importances.
    encoded_imp, grouped_imp = feature_importances(lgbm)
    logger.info("LightGBM importances by original feature (gain summed over one-hot columns):\n%s",
                grouped_imp.to_string(float_format=lambda v: f"{v:.1f}"))
    logger.info("Top 12 encoded columns by gain:\n%s",
                encoded_imp.head(12).to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    results = {
        "lgbm_grid": LGBM_GRID, "lgbm_best_params": best_params,
        "lgbm_search": search.to_dict(orient="records"),
        "thresholds": thresholds, "threshold_rules": rules,
        "validation": val_metrics,
        "assumed_costs": {"contact_cost": ASSUMED_CONTACT_COST, "saved_value": ASSUMED_SAVED_VALUE},
        "cost_sensitivity_val": costs.to_dict(orient="records"),
        "feature_importance_grouped": grouped_imp.reset_index().to_dict(orient="records"),
    }

    if skip_test:
        logger.info("--skip-test: test set NOT loaded; no test metrics, figures or metrics file written.")
    else:
        # The single evaluation on the untouched test set, thresholds frozen.
        X_test, y_test = load_split("test")
        logger.info("Test: %d rows. Evaluating each model ONCE at its frozen threshold.", len(X_test))
        test_metrics = {}
        test_scores = {name: scorers[name](X_test) for name in MODEL_NAMES}
        for name in MODEL_NAMES:
            pred = (test_scores[name] >= thresholds[name]).astype(int)
            m = evaluate(y_test, test_scores[name], pred)
            tp, fp = m["tp"], m["fp"]
            m["precision_at_threshold"] = tp / (tp + fp) if tp + fp else float("nan")
            m["recall_at_threshold"] = tp / (tp + m["fn"])
            m["net_benefit_assumed_costs"] = net_benefit(y_test, pred)
            test_metrics[name] = m
        test_table = pd.DataFrame(test_metrics).T.astype({"tp": int, "fp": int, "fn": int, "tn": int})
        test_table.insert(0, "threshold", pd.Series(thresholds))
        tcols = ["threshold", "pr_auc", "recall_at_precision", "f1_macro", "precision_at_threshold",
                 "recall_at_threshold", "tp", "fp", "fn", "tn", "net_benefit_assumed_costs"]
        logger.info("TEST metrics (thresholds frozen from validation):\n%s",
                    test_table[tcols].to_string(float_format=lambda v: f"{v:.4f}"))
        results["test"] = test_metrics

        # Uncertainty of the LightGBM vs logistic regression PR-AUC gap on test
        # (paired bootstrap: same resampled customers for both models).
        boot = paired_bootstrap_pr_auc(y_test, test_scores["lightgbm"], test_scores["logistic_regression"],
                                       n_boot=2000, seed=RANDOM_SEED)
        logger.info("Paired bootstrap, test PR-AUC (LightGBM - logistic regression), B=%d, seed=%d: "
                    "observed %+.4f, 95%% CI [%+.4f, %+.4f], bootstrap SD %.4f, LightGBM better in %.1f%% of resamples",
                    boot["n_boot"], boot["seed"], boot["observed_diff"], boot["ci95_low"], boot["ci95_high"],
                    boot["bootstrap_sd"], 100 * boot["share_resamples_a_better"])
        results["test_bootstrap_pr_auc_lgbm_minus_logreg"] = boot

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plot_pr_curves(
            y_test,
            {"LightGBM": test_scores["lightgbm"], "Logistic regression": test_scores["logistic_regression"],
             "Heuristic rule": test_scores["heuristic_rule"]},
            ("LightGBM", thresholds["lightgbm"]),
            FIGURES_DIR / "pr_curve_test.png", "Precision-recall curves, test set (n = %d)" % len(y_test))
        plot_confusion_matrix(y_test, (test_scores["lightgbm"] >= thresholds["lightgbm"]).astype(int),
                              FIGURES_DIR / "confusion_matrix_test.png",
                              "LightGBM, test set, threshold %.3f" % thresholds["lightgbm"])
        logger.info("Saved figures to %s", FIGURES_DIR.relative_to(REPO_ROOT))

        (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
        logger.info("Saved %s", (ARTIFACTS_DIR / "metrics.json").relative_to(REPO_ROOT))

    # Export: fitted pipeline (preprocessing + model) and its frozen threshold.
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    bundle = {
        "pipeline": lgbm,
        "threshold": thresholds["lightgbm"],
        "threshold_rule": rules["lightgbm"],
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "train_churn_rate": float(y_train.mean()),  # prior rate, available for a fallback
        # Heuristic rule as the first fallback layer: its validation-chosen score cut-off,
        # and the TRAINING churn rate for each rule score (0, 1, 2) as its probability.
        "heuristic": {
            "threshold": float(thresholds["heuristic_rule"]),
            "churn_rate_by_score": {int(k): float(v) for k, v in y_train.groupby(heuristic_score(X_train)).mean().items()},
        },
        # 20 validation rows and their training-time probabilities, so a test can check
        # that the loaded artifact reproduces them without needing any data files.
        "reference_predictions": {
            "inputs": X_val.head(20).to_dict(orient="records"),
            "probabilities": lgbm.predict_proba(X_val.head(20))[:, 1].tolist(),
        },
        # Observed training range of each numeric input; the API logs values outside it.
        "numeric_train_ranges": {f: [float(X_train[f].min()), float(X_train[f].max())] for f in NUMERIC_FEATURES},
        "lgbm_params": best_params,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "lightgbm": lightgbm.__version__, "numpy": np.__version__, "pandas": pd.__version__},
    }
    joblib.dump(bundle, MODEL_PATH)
    logger.info("Saved %s (threshold %.4f)", MODEL_PATH.relative_to(REPO_ROOT), thresholds["lightgbm"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-test", action="store_true", help="never load the test set (development run)")
    main(parser.parse_args().skip_test)
