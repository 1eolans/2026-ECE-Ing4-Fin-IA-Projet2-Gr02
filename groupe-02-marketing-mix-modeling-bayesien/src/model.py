import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def fit_bayesian_mmm(
    features: pd.DataFrame,
    sales: pd.Series,
    draws: int = 800,
    tune: int = 800,
    chains: int = 2,
    target_accept: float = 0.9,
    random_seed: int = 42,
):
    """Fit a simple Bayesian MMM with transformed media features."""
    x = features.to_numpy(dtype=float)
    y = sales.to_numpy(dtype=float)

    with pm.Model() as model:
        intercept = pm.Normal("intercept", mu=y.mean(), sigma=2 * y.std())
        beta = pm.HalfNormal("beta", sigma=200.0, shape=x.shape[1])
        sigma = pm.HalfNormal("sigma", sigma=y.std())

        mu = intercept + pm.math.dot(x, beta)
        pm.Normal("sales", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )

    return model, idata


def fit_hierarchical_bayesian_mmm(
    features: pd.DataFrame,
    sales: pd.Series,
    market_idx: np.ndarray,
    n_markets: int,
    draws: int = 500,
    tune: int = 500,
    chains: int = 2,
    target_accept: float = 0.92,
    random_seed: int = 42,
):
    """Fit a hierarchical Bayesian MMM with market-level random effects."""
    x = features.to_numpy(dtype=float)
    y = sales.to_numpy(dtype=float)
    market_idx = market_idx.astype(int)
    n_channels = x.shape[1]

    with pm.Model() as model:
        global_intercept = pm.Normal("global_intercept", mu=y.mean(), sigma=2 * y.std())
        sigma_market_intercept = pm.HalfNormal("sigma_market_intercept", sigma=40.0)
        market_intercept_offset = pm.Normal("market_intercept_offset", mu=0.0, sigma=1.0, shape=n_markets)
        market_intercept = pm.Deterministic(
            "market_intercept",
            global_intercept + market_intercept_offset * sigma_market_intercept,
        )

        # Log-normal hierarchical prior keeps media effects positive.
        log_beta_mu = pm.Normal("log_beta_mu", mu=np.log(90.0), sigma=1.0, shape=n_channels)
        log_beta_sigma = pm.HalfNormal("log_beta_sigma", sigma=0.6, shape=n_channels)
        log_beta_offset = pm.Normal(
            "log_beta_offset",
            mu=0.0,
            sigma=1.0,
            shape=(n_markets, n_channels),
        )
        beta_market = pm.Deterministic(
            "beta_market",
            pm.math.exp(log_beta_mu + log_beta_offset * log_beta_sigma),
        )

        sigma = pm.HalfNormal("sigma", sigma=y.std())

        mu = market_intercept[market_idx] + (beta_market[market_idx] * x).sum(axis=1)
        pm.Normal("sales", mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=False,
        )

    return model, idata


def posterior_summary(idata, channels: list[str]) -> pd.DataFrame:
    summary = az.summary(idata, var_names=["intercept", "beta", "sigma"], round_to=4)
    index = []
    for name in summary.index:
        if name.startswith("beta["):
            idx = int(name.split("[")[1].split("]")[0])
            index.append(f"beta_{channels[idx]}")
        else:
            index.append(name)
    summary.index = index
    return summary.reset_index(names="parameter")


def hierarchical_posterior_summary(idata) -> pd.DataFrame:
    summary = az.summary(
        idata,
        var_names=["global_intercept", "sigma_market_intercept", "log_beta_mu", "log_beta_sigma", "sigma"],
        round_to=4,
    )
    return summary.reset_index(names="parameter")


def posterior_mean_params(idata, channels: list[str]) -> tuple[float, dict[str, float]]:
    intercept_mean = float(idata.posterior["intercept"].mean().item())
    beta_mean = idata.posterior["beta"].mean(dim=("chain", "draw")).to_numpy()
    beta_map = {ch: float(beta_mean[i]) for i, ch in enumerate(channels)}
    return intercept_mean, beta_map


def hierarchical_posterior_means(
    idata,
    channels: list[str],
    market_names: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    beta_market = idata.posterior["beta_market"].mean(dim=("chain", "draw")).to_numpy()
    intercept_market = idata.posterior["market_intercept"].mean(dim=("chain", "draw")).to_numpy()

    rows = []
    for m_idx, market in enumerate(market_names):
        for c_idx, ch in enumerate(channels):
            rows.append(
                {
                    "market": market,
                    "channel": ch,
                    "beta_mean": float(beta_market[m_idx, c_idx]),
                }
            )

    market_beta_df = pd.DataFrame(rows)
    intercept_map = {market_names[i]: float(intercept_market[i]) for i in range(len(market_names))}
    return market_beta_df, intercept_map


def predict_sales(features: pd.DataFrame, intercept: float, beta_map: dict[str, float]) -> np.ndarray:
    mu = np.full(len(features), intercept, dtype=float)
    for ch in features.columns:
        mu += beta_map[ch] * features[ch].to_numpy(dtype=float)
    return mu


def predict_hierarchical_sales(
    features: pd.DataFrame,
    market_series: pd.Series,
    market_beta_df: pd.DataFrame,
    market_intercept_map: dict[str, float],
) -> np.ndarray:
    pred = np.zeros(len(features), dtype=float)
    channels = features.columns.tolist()

    beta_lookup = (
        market_beta_df.pivot(index="market", columns="channel", values="beta_mean")
        .reindex(columns=channels)
        .to_dict(orient="index")
    )

    for i, market in enumerate(market_series.to_numpy()):
        base = market_intercept_map[market]
        coefs = beta_lookup[market]
        value = base
        for ch in channels:
            value += coefs[ch] * float(features.iloc[i][ch])
        pred[i] = value

    return pred
