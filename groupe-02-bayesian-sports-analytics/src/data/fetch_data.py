"""
Data fetching module for Premier League match data.
Source: football-data.co.uk (free, no API key required)

Columns used:
    HomeTeam, AwayTeam  - team names
    FTHG, FTAG          - full-time home/away goals
    FTR                 - full-time result (H/D/A)
    B365H, B365D, B365A - Bet365 odds (home/draw/away)
    Date                - match date
"""

import io
import requests
import numpy as np
import pandas as pd

SEASON_URLS = {
    "2021-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
}

ODDS_COLS = ["B365H", "B365D", "B365A"]
CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def fetch_season(season: str) -> pd.DataFrame:
    """Download and parse one season of Premier League data."""
    url = SEASON_URLS[season]
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), encoding="latin-1")
    df["season"] = season
    # Keep only rows with complete match data
    df = df.dropna(subset=["FTHG", "FTAG"])
    return df


def fetch_all_seasons(seasons=None) -> pd.DataFrame:
    """Download multiple seasons and concatenate."""
    if seasons is None:
        seasons = list(SEASON_URLS.keys())
    frames = []
    for s in seasons:
        print(f"Fetching season {s}...")
        df = fetch_season(s)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the raw dataframe for modelling.
    Returns a dataframe with integer team indices and selected columns.
    """
    keep = CORE_COLS + [c for c in ODDS_COLS if c in df.columns] + ["season"]
    df = df[keep].copy()

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Integer indices for teams
    all_teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    df["home_idx"] = df["HomeTeam"].map(team_to_idx)
    df["away_idx"] = df["AwayTeam"].map(team_to_idx)

    # Match week proxy (rank by date within season)
    df["matchweek"] = df.groupby("season")["Date"].rank(method="dense").astype(int)

    # Implied probabilities from Bet365 odds (margin-adjusted)
    if all(c in df.columns for c in ODDS_COLS):
        raw_p = 1.0 / df[ODDS_COLS].values          # shape (N, 3)
        margin = raw_p.sum(axis=1, keepdims=True)
        adj = raw_p / margin                          # remove bookmaker margin
        df[["prob_home", "prob_draw", "prob_away"]] = adj

    return df, all_teams, team_to_idx


def train_test_split_by_matchweek(df: pd.DataFrame, test_matchweeks: int = 5):
    """
    Hold out the last `test_matchweeks` matchweeks of the most recent season
    as an out-of-sample test set (simulates live prediction).
    """
    last_season = df["season"].max()
    mask_last = df["season"] == last_season
    max_mw = df.loc[mask_last, "matchweek"].max()
    cutoff = max_mw - test_matchweeks

    train = df[~(mask_last & (df["matchweek"] > cutoff))].copy()
    test = df[mask_last & (df["matchweek"] > cutoff)].copy()
    return train, test


if __name__ == "__main__":
    raw = fetch_all_seasons(["2022-23", "2023-24", "2024-25"])
    df, teams, _ = prepare_dataset(raw)
    train, test = train_test_split_by_matchweek(df)
    print(f"Teams: {len(teams)}")
    print(f"Train matches: {len(train)}, Test matches: {len(test)}")
    print(df.head())
