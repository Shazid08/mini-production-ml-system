"""Evaluation metrics shared by the baselines, the final model and the report.

Metrics (churn = positive class, about 26.5% of customers):
  - PR-AUC (primary): area under the precision-recall curve, computed as
    average precision. Threshold-free; a no-skill model scores the churn rate.
  - Recall at a target precision: the share of churners caught at the best
    threshold that still keeps precision at or above the target.
  - F1-macro: the plain average of the F1 score for "churn" and "no churn",
    so the large majority class cannot hide a poor minority-class result.
"""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_recall_curve

# Business assumption (not a measured fact): a retention campaign is acceptable
# if at least 60% of the customers it contacts are true churners.
TARGET_PRECISION = 0.60


def pr_auc(y_true, scores) -> float:
    """Average precision: precision averaged over every recall level reached."""
    return float(average_precision_score(y_true, scores))


def recall_at_precision(y_true, scores, min_precision: float = TARGET_PRECISION):
    """Highest recall among thresholds whose precision is >= min_precision.

    Returns (recall, threshold). If no threshold reaches the target precision
    the recall is 0.0 and the threshold is None.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall have one more entry than thresholds: the final point
    # (precision=1, recall=0) is added by sklearn and has no threshold.
    meets_target = precision[:-1] >= min_precision
    if not meets_target.any():
        return 0.0, None
    best = int(np.argmax(np.where(meets_target, recall[:-1], -1.0)))
    return float(recall[best]), float(thresholds[best])


def f1_macro(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def evaluate(y_true, scores, y_pred, min_precision: float = TARGET_PRECISION) -> dict:
    """All metrics for one model: threshold-free ones from `scores`,
    threshold-based ones (F1-macro, confusion matrix) from `y_pred`."""
    recall, threshold = recall_at_precision(y_true, scores, min_precision)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": pr_auc(y_true, scores),
        "recall_at_precision": recall,
        "target_precision": min_precision,
        "threshold_at_target_precision": threshold,
        "f1_macro": f1_macro(y_true, y_pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ---------------------------------------------------------------- threshold choice
def choose_threshold(y_true, scores, min_precision: float = TARGET_PRECISION):
    """Threshold rule used for every model (fixed before seeing results):

    1. Highest recall with precision >= min_precision (ties -> lowest threshold).
    2. If no threshold reaches that precision, the threshold with the highest
       F1-macro. "Flag nobody" (threshold = +inf) is one of the candidates.

    A record is flagged as churn when score >= threshold.
    Returns (threshold, rule_used).
    """
    _, threshold = recall_at_precision(y_true, scores, min_precision)
    if threshold is not None:
        return threshold, f"max recall with precision >= {min_precision:.2f}"
    scores = np.asarray(scores, dtype=float)
    candidates = np.append(np.unique(scores), np.inf)
    f1s = [f1_macro(y_true, (scores >= t).astype(int)) for t in candidates]
    return float(candidates[int(np.argmax(f1s))]), "fallback: max F1-macro (target precision unreachable)"


# ---------------------------------------------------------------- business cost
# ASSUMED, illustrative values, not measured from data:
ASSUMED_CONTACT_COST = 50.0   # cost of contacting one customer (agent time + incentive)
ASSUMED_SAVED_VALUE = 300.0   # expected value kept when a true churner is contacted


def net_benefit(y_true, y_pred, contact_cost: float = ASSUMED_CONTACT_COST,
                saved_value: float = ASSUMED_SAVED_VALUE) -> float:
    """Net benefit of a retention campaign that contacts every flagged customer:
    saved_value per true churner reached, minus contact_cost per contact."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    contacted = int((y_pred == 1).sum())
    return saved_value * tp - contact_cost * contacted


def best_cost_threshold(y_true, scores, contact_cost: float, saved_value: float):
    """Threshold that maximises net benefit (candidates: every distinct score,
    plus +inf = contact nobody). Returns (threshold, net_benefit)."""
    scores = np.asarray(scores, dtype=float)
    candidates = np.append(np.unique(scores), np.inf)
    benefits = [net_benefit(y_true, (scores >= t).astype(int), contact_cost, saved_value) for t in candidates]
    best = int(np.argmax(benefits))
    return float(candidates[best]), float(benefits[best])


# ---------------------------------------------------------------- figures
# Colours: first three slots of a colour-blind-checked categorical palette,
# plus neutral greys for text and the no-skill reference line.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID_GREY = "#0b0b0b", "#52514e", "#e4e3df"


def _style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(color=GRID_GREY, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_pr_curves(y_true, curves: dict, operating_point: tuple, path, title: str) -> None:
    """Precision-recall curves for up to three models plus the no-skill line.

    curves: {label: scores}; operating_point: (label, threshold) to mark.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=200)
    _style_axes(ax)
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, color=TEXT_SECONDARY, linestyle="--", linewidth=1.2,
               label=f"Dummy / no skill (AP = {prevalence:.3f})")
    for colour, (label, scores) in zip(SERIES_COLOURS, curves.items()):
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax.plot(recall, precision, color=colour, linewidth=2, drawstyle="steps-post",
                label=f"{label} (AP = {pr_auc(y_true, scores):.3f})")
        if label == operating_point[0]:
            pred = np.asarray(scores) >= operating_point[1]
            tp = int((pred & (np.asarray(y_true) == 1)).sum())
            p, r = tp / max(int(pred.sum()), 1), tp / int(np.sum(y_true))
            ax.plot([r], [p], marker="o", markersize=8, color=colour,
                    markeredgecolor="white", markeredgewidth=2, zorder=5)
            ax.annotate(f"chosen threshold {operating_point[1]:.3f}\nP = {p:.2f}, R = {r:.2f}",
                        (r, p), xytext=(12, 10), textcoords="offset points",
                        fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Recall (share of churners caught)", color=TEXT_PRIMARY, fontsize=10)
    ax.set_ylabel("Precision (share of flags that churn)", color=TEXT_PRIMARY, fontsize=10)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, path, title: str) -> None:
    """2x2 confusion matrix, single blue ramp, counts and row shares in each cell."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ["#eef4fc", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(4.4, 3.8), dpi=200)
    ax.imshow(cm, cmap=cmap)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / cm[i].sum()
            dark = cm[i, j] > cm.max() * 0.6
            ax.text(j, i, f"{cm[i, j]}\n({share:.0%} of row)", ha="center", va="center",
                    fontsize=10, color="white" if dark else TEXT_PRIMARY)
    ax.set_xticks([0, 1], ["Predicted: stay", "Predicted: churn"])
    ax.set_yticks([0, 1], ["Actual: stay", "Actual: churn"])
    ax.tick_params(colors=TEXT_PRIMARY, labelsize=9, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- uncertainty
def paired_bootstrap_pr_auc(y_true, scores_a, scores_b, n_boot: int = 2000, seed: int = 42) -> dict:
    """Paired bootstrap of the PR-AUC difference (model A minus model B).

    Each resample draws len(y_true) customers with replacement; the SAME draw is
    used for both models, so the comparison is paired. Returns the observed
    difference, a 95% percentile interval and the share of resamples where A wins.
    """
    y = np.asarray(y_true)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = pr_auc(y[idx], a[idx]) - pr_auc(y[idx], b[idx])
    return {
        "observed_diff": pr_auc(y, a) - pr_auc(y, b),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "share_resamples_a_better": float((diffs > 0).mean()),
        "bootstrap_sd": float(diffs.std(ddof=1)),
        "n_boot": n_boot, "seed": seed,
    }
