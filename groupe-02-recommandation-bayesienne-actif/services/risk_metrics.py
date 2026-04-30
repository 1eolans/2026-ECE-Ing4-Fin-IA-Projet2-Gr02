import numpy as np
from scipy.stats import norm


def annualized_return(daily_returns):
    return float(np.mean(daily_returns) * 252)


def annualized_volatility(daily_returns):
    return float(np.std(daily_returns, ddof=1) * np.sqrt(252))


def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    ann_ret = annualized_return(daily_returns)
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def var_parametric(daily_returns, alpha=0.05):
    
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns, ddof=1)
    return float(norm.ppf(alpha, loc=mu, scale=sigma))


def cvar_parametric(daily_returns, alpha=0.05):
    
    mu = np.mean(daily_returns)
    sigma = np.std(daily_returns, ddof=1)
    z = norm.ppf(alpha)
    cvar = mu - sigma * (norm.pdf(z) / alpha)
    return float(cvar)


def summarize_asset_risk(returns_df, alpha=0.05):
    rows = []
    for col in returns_df.columns:
        series = returns_df[col].dropna().values
        rows.append({
            "Ticker": col,
            "AnnualReturn": annualized_return(series),
            "AnnualVolatility": annualized_volatility(series),
            "Sharpe": sharpe_ratio(series),
            "VaR": var_parametric(series, alpha),
            "CVaR": cvar_parametric(series, alpha),
        })
    return rows