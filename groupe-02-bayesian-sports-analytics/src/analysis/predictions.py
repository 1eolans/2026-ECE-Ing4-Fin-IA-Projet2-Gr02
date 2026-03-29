"""
Evaluation metrics for Bayesian match predictions.

Metrics computed:
    - Accuracy          : predicted result matches actual result
    - Log-loss (RPS)    : Ranked Probability Score (proper scoring rule for ordinal outcomes)
    - Brier score       : mean squared error of win/draw/loss probabilities
    - Calibration       : reliability diagram data
"""

import numpy as np
import pandas as pd


def result_from_goals(home: float, away: float) -> str:
    if home > away:
        return "H"
    elif home == away:
        return "D"
    return "A"


def ranked_probability_score(p_home: float, p_draw: float, p_away: float, actual: str) -> float:
    """
    Ranked Probability Score for a 3-outcome ordinal event (H < D < A ranking by cumulative).
    Lower is better.
    """
    probs = np.array([p_home, p_draw, p_away])
    outcomes = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}
    obs = np.array(outcomes[actual])
    cumprob = np.cumsum(probs)
    cumobs  = np.cumsum(obs)
    return np.mean((cumprob[:-1] - cumobs[:-1]) ** 2)


def evaluate_predictions(pred_df: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics on a prediction dataframe.

    Expected columns: p_home, p_draw, p_away, actual_result
    """
    pred_result = pred_df.apply(
        lambda r: result_from_goals(r["pred_home_goals"], r["pred_away_goals"]),
        axis=1,
    )
    accuracy = (pred_result == pred_df["actual_result"]).mean()

    rps_scores = pred_df.apply(
        lambda r: ranked_probability_score(r["p_home"], r["p_draw"], r["p_away"], r["actual_result"]),
        axis=1,
    )
    avg_rps = rps_scores.mean()

    # Brier score: MSE of outcome indicator
    def brier(row):
        probs = np.array([row["p_home"], row["p_draw"], row["p_away"]])
        obs   = {"H": [1, 0, 0], "D": [0, 1, 0], "A": [0, 0, 1]}[row["actual_result"]]
        return np.mean((probs - obs) ** 2)

    brier_score = pred_df.apply(brier, axis=1).mean()

    return {
        "accuracy": accuracy,
        "avg_rps": avg_rps,
        "brier_score": brier_score,
        "n_matches": len(pred_df),
    }


def calibration_data(pred_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """
    Compute calibration (reliability diagram) data for the home-win probability.
    Groups predictions into bins and compares predicted vs observed frequency.
    """
    df = pred_df.copy()
    df["actual_home_win"] = (df["actual_result"] == "H").astype(float)
    df["bin"] = pd.cut(df["p_home"], bins=n_bins, labels=False)

    cal = df.groupby("bin").agg(
        pred_prob=("p_home", "mean"),
        obs_freq=("actual_home_win", "mean"),
        count=("actual_home_win", "count"),
    ).reset_index()
    return cal


def confusion_matrix_results(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Return a 3×3 confusion matrix (actual vs predicted result)."""
    pred_df = pred_df.copy()
    pred_df["pred_result"] = pred_df.apply(
        lambda r: result_from_goals(r["pred_home_goals"], r["pred_away_goals"]), axis=1
    )
    labels = ["H", "D", "A"]
    matrix = pd.crosstab(
        pred_df["actual_result"],
        pred_df["pred_result"],
        rownames=["Actual"],
        colnames=["Predicted"],
    ).reindex(index=labels, columns=labels, fill_value=0)
    return matrix
