import matplotlib
matplotlib.use("Agg")         
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import PLOT_MONTH, RESULTS_DIR


def score_table(preds):
    """preds has a column 'actual' plus one column per method. Returns (scores, scored_days).
    Only days where EVERY method has a prediction are scored, so all methods are judged on the same days."""
    scored = preds.dropna()
    errors = scored.drop(columns="actual").sub(scored["actual"], axis=0)      # prediction minus actual

    scores = pd.DataFrame({
        "MAE_kwh": errors.abs().mean(),
        "RMSE_kwh": np.sqrt((errors ** 2).mean()),
    }).round(2)

    baselines = [m for m in scores.index if m.startswith("baseline")]
    best_baseline = scores.loc[baselines, "MAE_kwh"].idxmin()
    scores["better_than_baseline_2_%"] = ((1 - scores["MAE_kwh"] / scores.loc["baseline_2", "MAE_kwh"]) * 100).round(0)
    scores["better_than_best_baseline_%"] = ((1 - scores["MAE_kwh"] / scores.loc[best_baseline, "MAE_kwh"]) * 100).round(0)
    scores["days"] = len(scored)
    scores.attrs["best_baseline"] = best_baseline
    return scores, scored


def save_results(scores, scored):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS_DIR / "valid_scores.csv", index_label="method")
    scored.to_csv(RESULTS_DIR / "valid_predictions.csv", index_label="date")
    (RESULTS_DIR / "valid_scores.txt").write_text(scores.to_string())


def plot_comparison(scores, preds, path=None):
    methods = ["baseline_2", "linear_regression", "random_forest", "gradient_boosting"]
    target = 0.7 * scores.loc["baseline_2", "MAE_kwh"]                          # 30% better than baseline 2

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    scores.loc[methods, ["MAE_kwh", "RMSE_kwh"]].plot(kind="bar", ax=ax[0], rot=15)
    ax[0].axhline(target, color="red", linestyle="--", label=f"target: {target:.1f} kWh (30% better than baseline 2)")
    ax[0].set_title("Error on the validation year (lower is better)")
    ax[0].set_ylabel("kWh per day")
    ax[0].set_ylim(0, scores.loc[methods, ["MAE_kwh", "RMSE_kwh"]].max().max() * 1.35)   # room for the legend
    ax[0].legend(fontsize=9)

    preds.loc[PLOT_MONTH, ["actual"] + methods].plot(ax=ax[1], title=f"{PLOT_MONTH}: actual vs predicted")
    ax[1].set_ylabel("kWh per day")

    plt.tight_layout()
    plt.savefig(path or RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
