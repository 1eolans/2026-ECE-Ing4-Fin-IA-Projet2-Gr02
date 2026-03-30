import pandas as pd
import yfinance as yf


def load_prices(tickers, period="3y", interval="1d"):
    
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if data.empty:
        raise ValueError("Aucune donnée téléchargée. Vérifie les tickers ou la connexion internet.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            prices = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        prices = data.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    prices = prices.dropna(how="all")
    return prices


def compute_returns(prices):
    
    returns = prices.pct_change().dropna()
    if returns.empty:
        raise ValueError("Impossible de calculer les rendements : données insuffisantes.")
    return returns


def load_returns(tickers, period="3y", interval="1d"):
   
    prices = load_prices(tickers, period=period, interval=interval)
    returns = compute_returns(prices)
    return prices, returns