import numpy as np
import pandas as pd

from features import build_mmm_features, build_multimarket_mmm_features


def generate_synthetic_mmm_data(n_weeks: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic weekly sales and media spends (single market)."""
    rng = np.random.default_rng(seed)
    weeks = np.arange(n_weeks)

    spend_tv = rng.normal(loc=180.0, scale=35.0, size=n_weeks).clip(30, None)
    spend_facebook = rng.normal(loc=90.0, scale=20.0, size=n_weeks).clip(10, None)
    spend_google = rng.normal(loc=110.0, scale=25.0, size=n_weeks).clip(15, None)

    raw = pd.DataFrame(
        {
            "week": weeks,
            "spend_tv": spend_tv,
            "spend_facebook": spend_facebook,
            "spend_google": spend_google,
        }
    )

    channels = ["tv", "facebook", "google"]
    alpha = {"tv": 0.55, "facebook": 0.35, "google": 0.45}
    ec50 = {"tv": 220.0, "facebook": 90.0, "google": 120.0}
    slope = {"tv": 1.8, "facebook": 2.2, "google": 2.0}

    transformed = build_mmm_features(
        raw,
        channels=channels,
        adstock_alpha=alpha,
        hill_ec50=ec50,
        hill_slope=slope,
    )

    base = 550.0
    trend = 0.7 * weeks
    seasonal = 20.0 * np.sin(2 * np.pi * weeks / 52)

    true_betas = {"tv": 120.0, "facebook": 80.0, "google": 95.0}

    signal = (
        base
        + trend
        + seasonal
        + true_betas["tv"] * transformed["tv"].to_numpy()
        + true_betas["facebook"] * transformed["facebook"].to_numpy()
        + true_betas["google"] * transformed["google"].to_numpy()
    )

    noise = rng.normal(0.0, 12.0, size=n_weeks)
    sales = signal + noise

    raw["sales"] = sales
    return raw


def generate_synthetic_multimarket_mmm_data(
    n_markets: int = 4,
    n_weeks: int = 130,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic MMM data for multiple markets with heterogeneous effects."""
    rng = np.random.default_rng(seed)
    channels = ["tv", "facebook", "google"]

    rows = []
    market_names = [f"market_{i+1}" for i in range(n_markets)]

    base_beta = np.array([110.0, 85.0, 100.0], dtype=float)
    base_intercept = 540.0

    for market in market_names:
        weeks = np.arange(n_weeks)

        # Market-specific multipliers introduce heterogeneity.
        spend_multiplier = rng.uniform(0.75, 1.30)
        beta_multiplier = rng.uniform(0.75, 1.25, size=len(channels))
        intercept_shift = rng.normal(0.0, 30.0)

        spend_tv = (
            rng.normal(loc=180.0 * spend_multiplier, scale=35.0, size=n_weeks)
            .clip(25, None)
            .astype(float)
        )
        spend_facebook = (
            rng.normal(loc=90.0 * spend_multiplier, scale=20.0, size=n_weeks)
            .clip(10, None)
            .astype(float)
        )
        spend_google = (
            rng.normal(loc=110.0 * spend_multiplier, scale=25.0, size=n_weeks)
            .clip(15, None)
            .astype(float)
        )

        market_df = pd.DataFrame(
            {
                "market": market,
                "week": weeks,
                "spend_tv": spend_tv,
                "spend_facebook": spend_facebook,
                "spend_google": spend_google,
            }
        )

        rows.append((market_df, beta_multiplier, intercept_shift))

    raw = pd.concat([r[0] for r in rows], ignore_index=True)

    alpha = {"tv": 0.50, "facebook": 0.35, "google": 0.45}
    ec50 = {"tv": 220.0, "facebook": 90.0, "google": 120.0}
    slope = {"tv": 1.8, "facebook": 2.2, "google": 2.0}

    transformed = build_multimarket_mmm_features(
        raw,
        channels=channels,
        adstock_alpha=alpha,
        hill_ec50=ec50,
        hill_slope=slope,
        market_col="market",
        week_col="week",
    )

    sales = np.zeros(len(raw), dtype=float)
    market_to_idx = {name: i for i, name in enumerate(market_names)}

    for market_name, (_, beta_multiplier, intercept_shift) in zip(market_names, rows):
        mask = raw["market"] == market_name
        idx = np.where(mask.to_numpy())[0]

        w = raw.loc[mask, "week"].to_numpy(dtype=float)
        trend = 0.6 * w
        seasonal = 25.0 * np.sin(2 * np.pi * w / 52 + market_to_idx[market_name] * 0.2)

        betas = base_beta * beta_multiplier
        signal = (
            base_intercept
            + intercept_shift
            + trend
            + seasonal
            + betas[0] * transformed.loc[mask, "tv"].to_numpy()
            + betas[1] * transformed.loc[mask, "facebook"].to_numpy()
            + betas[2] * transformed.loc[mask, "google"].to_numpy()
        )
        noise = rng.normal(0.0, 14.0, size=len(idx))
        sales[idx] = signal + noise

    raw["sales"] = sales
    return raw
