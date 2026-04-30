"""
Value bet identification — comparing Bayesian model probabilities to bookmaker odds.

A value bet exists when our model believes the true probability of an outcome
is higher than what the bookmaker's odds imply (after margin removal).

Expected Value: EV = p_model * odds_decimal - 1
A bet has positive EV when EV > 0.

Kelly Criterion: fraction of bankroll to bet = (p * odds - 1) / (odds - 1)
"""

import numpy as np
import pandas as pd


def implied_prob(odds: float) -> float:
    """Convert decimal odds to raw implied probability."""
    return 1.0 / odds if odds > 0 else np.nan


def margin_adjusted_prob(odds_h: float, odds_d: float, odds_a: float):
    """Remove bookmaker margin and return fair probabilities."""
    raw = np.array([implied_prob(odds_h), implied_prob(odds_d), implied_prob(odds_a)])
    if np.any(np.isnan(raw)):
        return np.array([np.nan, np.nan, np.nan])
    return raw / raw.sum()


def expected_value(p_model: float, odds_decimal: float) -> float:
    """EV of a 1-unit bet at decimal odds given model probability p_model."""
    return p_model * odds_decimal - 1.0


def kelly_fraction(p_model: float, odds_decimal: float) -> float:
    """Kelly criterion bet size as fraction of bankroll."""
    if odds_decimal <= 1.0:
        return 0.0
    f = (p_model * odds_decimal - 1.0) / (odds_decimal - 1.0)
    return max(0.0, f)


def find_value_bets(
    pred_df: pd.DataFrame,
    min_ev: float = 0.05,
    max_kelly: float = 0.25,
) -> pd.DataFrame:
    """
    Identify value bets in pred_df.

    pred_df must contain:
        p_home, p_draw, p_away       - model probabilities
        B365H, B365D, B365A          - Bet365 decimal odds
        HomeTeam, AwayTeam, Date     - match info
        actual_result                - actual outcome (optional, for backtesting)

    Parameters
    ----------
    min_ev    : minimum expected value threshold (default 5%)
    max_kelly : cap on Kelly fraction to avoid ruin (default 25%)

    Returns
    -------
    DataFrame of value bets sorted by EV descending
    """
    required_odds = ["B365H", "B365D", "B365A"]
    if not all(c in pred_df.columns for c in required_odds):
        raise ValueError("pred_df must contain B365H, B365D, B365A columns")

    rows = []
    for _, row in pred_df.iterrows():
        for outcome, p_col, odds_col in [
            ("H", "p_home", "B365H"),
            ("D", "p_draw", "B365D"),
            ("A", "p_away", "B365A"),
        ]:
            odds = row[odds_col]
            p    = row[p_col]
            if pd.isna(odds) or pd.isna(p):
                continue
            ev = expected_value(p, odds)
            kf = kelly_fraction(p, odds)
            if ev >= min_ev:
                entry = {
                    "Date":      row.get("Date", np.nan),
                    "HomeTeam":  row.get("HomeTeam", ""),
                    "AwayTeam":  row.get("AwayTeam", ""),
                    "bet_on":    outcome,
                    "odds":      odds,
                    "p_model":   p,
                    "p_implied": implied_prob(odds),
                    "edge":      p - implied_prob(odds),
                    "ev":        ev,
                    "kelly":     min(kf, max_kelly),
                }
                if "actual_result" in row:
                    entry["won"] = (row["actual_result"] == outcome)
                rows.append(entry)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("ev", ascending=False).reset_index(drop=True)


def backtest_strategy(
    value_bets: pd.DataFrame,
    initial_bankroll: float = 1000.0,
    bet_fraction: str = "kelly",   # "kelly" or "flat"
    flat_stake: float = 10.0,
) -> pd.DataFrame:
    """
    Simulate P&L of betting on all identified value bets.

    Parameters
    ----------
    bet_fraction : "kelly" uses fractional Kelly sizing; "flat" uses flat_stake
    """
    if value_bets.empty or "won" not in value_bets.columns:
        raise ValueError("value_bets must contain 'won' column (requires actual results)")

    bankroll = initial_bankroll
    history = []

    for _, bet in value_bets.sort_values("Date").iterrows():
        stake = (bankroll * bet["kelly"]) if bet_fraction == "kelly" else flat_stake
        stake = min(stake, bankroll)          # can't bet more than you have
        profit = stake * (bet["odds"] - 1) if bet["won"] else -stake
        bankroll += profit
        history.append({
            "Date":     bet["Date"],
            "match":    f"{bet['HomeTeam']} vs {bet['AwayTeam']}",
            "bet_on":   bet["bet_on"],
            "odds":     bet["odds"],
            "ev":       bet["ev"],
            "stake":    stake,
            "profit":   profit,
            "bankroll": bankroll,
            "won":      bet["won"],
        })

    return pd.DataFrame(history)


def summary_stats(backtest_df: pd.DataFrame) -> dict:
    """Return high-level summary of a backtest."""
    if backtest_df.empty:
        return {}
    return {
        "n_bets":        len(backtest_df),
        "win_rate":      backtest_df["won"].mean(),
        "total_profit":  backtest_df["profit"].sum(),
        "roi":           backtest_df["profit"].sum() / backtest_df["stake"].sum(),
        "max_drawdown":  (backtest_df["bankroll"].cummax() - backtest_df["bankroll"]).max(),
        "final_bankroll": backtest_df["bankroll"].iloc[-1],
    }
