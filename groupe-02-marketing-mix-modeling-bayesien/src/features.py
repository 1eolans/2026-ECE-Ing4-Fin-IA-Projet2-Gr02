import numpy as np
import pandas as pd


def geometric_adstock(spend: np.ndarray, alpha: float) -> np.ndarray:
    """Apply geometric adstock to a spend time series."""
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1).")

    out = np.zeros_like(spend, dtype=float)
    carry = 0.0
    for i, value in enumerate(spend):
        carry = value + alpha * carry
        out[i] = carry
    return out


def hill_transform(x: np.ndarray, ec50: float, slope: float) -> np.ndarray:
    """Saturation transform used in MMM."""
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)
    if ec50 <= 0:
        raise ValueError("ec50 must be > 0")
    if slope <= 0:
        raise ValueError("slope must be > 0")
    return 1.0 / (1.0 + (x / ec50) ** (-slope))


def build_mmm_features(
    df: pd.DataFrame,
    channels: list[str],
    adstock_alpha: dict[str, float],
    hill_ec50: dict[str, float],
    hill_slope: dict[str, float],
) -> pd.DataFrame:
    """Create transformed channel features for MMM (single market)."""
    transformed = pd.DataFrame(index=df.index)
    for ch in channels:
        spend = df[f"spend_{ch}"].to_numpy(dtype=float)
        adstocked = geometric_adstock(spend, alpha=adstock_alpha[ch])
        transformed[ch] = hill_transform(
            adstocked,
            ec50=hill_ec50[ch],
            slope=hill_slope[ch],
        )
    return transformed


def build_multimarket_mmm_features(
    df: pd.DataFrame,
    channels: list[str],
    adstock_alpha: dict[str, float],
    hill_ec50: dict[str, float],
    hill_slope: dict[str, float],
    market_col: str = "market",
    week_col: str = "week",
) -> pd.DataFrame:
    """Create transformed channel features by market with independent adstock memory."""
    transformed = pd.DataFrame(index=df.index)

    for market_name, market_df in df.groupby(market_col):
        market_idx = market_df.sort_values(week_col).index

        for ch in channels:
            spend = df.loc[market_idx, f"spend_{ch}"].to_numpy(dtype=float)
            adstocked = geometric_adstock(spend, alpha=adstock_alpha[ch])
            transformed.loc[market_idx, ch] = hill_transform(
                adstocked,
                ec50=hill_ec50[ch],
                slope=hill_slope[ch],
            )

    return transformed.sort_index()
