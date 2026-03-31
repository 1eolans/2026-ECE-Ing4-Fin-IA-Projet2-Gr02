import numpy as np
import pandas as pd


def estimate_channel_roi(
    raw_df: pd.DataFrame,
    transformed_features: pd.DataFrame,
    beta_map: dict[str, float],
    channels: list[str],
) -> pd.DataFrame:
    """Estimate channel ROI proxy from posterior mean betas."""
    rows = []
    for ch in channels:
        contrib = beta_map[ch] * transformed_features[ch].to_numpy(dtype=float)
        total_contrib = float(contrib.sum())
        total_spend = float(raw_df[f"spend_{ch}"].sum())
        roi = total_contrib / max(total_spend, 1e-9)
        rows.append(
            {
                "channel": ch,
                "total_spend": total_spend,
                "total_contribution": total_contrib,
                "roi_proxy": roi,
            }
        )
    return pd.DataFrame(rows).sort_values("roi_proxy", ascending=False)


def estimate_channel_roi_multimarket(
    raw_df: pd.DataFrame,
    transformed_features: pd.DataFrame,
    market_beta_df: pd.DataFrame,
    channels: list[str],
    market_col: str = "market",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate contributions and ROI by channel and market for hierarchical MMM."""
    beta_lookup = (
        market_beta_df.pivot(index="market", columns="channel", values="beta_mean")
        .reindex(columns=channels)
        .to_dict(orient="index")
    )

    contrib_rows = []
    for idx, row in raw_df.iterrows():
        market = row[market_col]
        for ch in channels:
            contrib_rows.append(
                {
                    "market": market,
                    "week": int(row["week"]),
                    "channel": ch,
                    "contribution": float(beta_lookup[market][ch] * transformed_features.loc[idx, ch]),
                    "spend": float(row[f"spend_{ch}"]),
                }
            )

    contrib_df = pd.DataFrame(contrib_rows)

    agg = (
        contrib_df.groupby(["market", "channel"], as_index=False)
        .agg(total_spend=("spend", "sum"), total_contribution=("contribution", "sum"))
    )
    agg["roi_proxy"] = agg["total_contribution"] / agg["total_spend"].clip(lower=1e-9)

    global_roi = (
        agg.groupby("channel", as_index=False)
        .agg(total_spend=("total_spend", "sum"), total_contribution=("total_contribution", "sum"))
    )
    global_roi["roi_proxy"] = global_roi["total_contribution"] / global_roi["total_spend"].clip(lower=1e-9)
    global_roi = global_roi.sort_values("roi_proxy", ascending=False)

    return agg, global_roi


def recommend_budget(
    roi_df: pd.DataFrame,
    total_budget: float,
    min_share: float = 0.1,
) -> pd.DataFrame:
    """Simple budget split recommendation based on ROI weights."""
    if total_budget <= 0:
        raise ValueError("total_budget must be > 0")

    channels = roi_df["channel"].tolist()
    n = len(channels)
    if not 0 <= min_share <= 1 / n:
        raise ValueError("min_share must be in [0, 1/n_channels]")

    roi = roi_df["roi_proxy"].to_numpy(dtype=float)
    roi = np.maximum(roi, 1e-9)
    weights = roi / roi.sum()

    base = np.full(n, min_share)
    remaining = 1.0 - base.sum()
    shares = base + remaining * weights

    out = roi_df.copy()
    out["recommended_share"] = shares
    out["recommended_budget"] = shares * total_budget
    return out.sort_values("recommended_budget", ascending=False)
