import numpy as np
import pandas as pd
from models.bayesian_recommender import BayesianRecommender


class Backtester:
    def __init__(self, returns_df, window=252, top_n=3, risk_aversion=1.0):
        self.returns_df = returns_df
        self.window = window
        self.top_n = top_n
        self.risk_aversion = risk_aversion

    def run(self):
        portfolio_returns = []
        dates = []

        if len(self.returns_df) <= self.window + 1:
            raise ValueError("Pas assez de données pour le backtest avec cette fenêtre.")

        for t in range(self.window, len(self.returns_df) - 1):
            train = self.returns_df.iloc[t - self.window:t]
            next_day = self.returns_df.iloc[t + 1]

            model = BayesianRecommender(train)
            model.fit()
            reco = model.recommend(risk_aversion=self.risk_aversion, top_n=self.top_n)

            selected_assets = reco["Ticker"].tolist()
            day_return = float(next_day[selected_assets].mean())

            portfolio_returns.append(day_return)
            dates.append(self.returns_df.index[t + 1])

        result = pd.DataFrame({
            "Date": dates,
            "PortfolioReturn": portfolio_returns
        })
        result["CumulativeReturn"] = (1 + result["PortfolioReturn"]).cumprod() - 1
        return result