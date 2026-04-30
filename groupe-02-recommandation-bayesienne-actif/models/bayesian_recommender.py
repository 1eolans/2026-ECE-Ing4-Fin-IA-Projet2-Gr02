import numpy as np
import pandas as pd


class BayesianRecommender:
    

    def __init__(self, returns_df, prior_mean=0.0005, prior_variance=1e-4):
        self.returns_df = returns_df.copy()
        self.assets = list(returns_df.columns)
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.posterior_stats = None

    @staticmethod
    def _posterior_normal_normal(sample, mu0, tau0_sq):
        sample = np.asarray(sample)
        sample = sample[~np.isnan(sample)]

        n = len(sample)
        if n == 0:
            return {
                "posterior_mean": mu0,
                "posterior_variance": tau0_sq,
                "sample_mean": np.nan,
                "sample_variance": np.nan,
                "n_obs": 0
            }

        x_bar = np.mean(sample)

        if n == 1:
            sigma_sq = max(np.var(sample), 1e-6)
        else:
            sigma_sq = max(np.var(sample, ddof=1), 1e-6)

        precision_prior = 1.0 / tau0_sq
        precision_likelihood = n / sigma_sq
        tau_n_sq = 1.0 / (precision_prior + precision_likelihood)
        mu_n = tau_n_sq * (precision_prior * mu0 + precision_likelihood * x_bar)

        return {
            "posterior_mean": float(mu_n),
            "posterior_variance": float(tau_n_sq),
            "sample_mean": float(x_bar),
            "sample_variance": float(sigma_sq),
            "n_obs": int(n)
        }

    def fit(self):
        rows = []
        for asset in self.assets:
            stats = self._posterior_normal_normal(
                self.returns_df[asset].dropna().values,
                mu0=self.prior_mean,
                tau0_sq=self.prior_variance
            )
            posterior_std = np.sqrt(stats["posterior_variance"])
            sample_std = np.sqrt(stats["sample_variance"]) if not np.isnan(stats["sample_variance"]) else np.nan

            rows.append({
                "Ticker": asset,
                "PosteriorMean": stats["posterior_mean"],
                "PosteriorStd": float(posterior_std),
                "SampleMean": stats["sample_mean"],
                "SampleStd": float(sample_std) if not np.isnan(sample_std) else np.nan,
                "Observations": stats["n_obs"]
            })

        self.posterior_stats = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
        return self.posterior_stats

    def recommend(self, risk_aversion=1.0, top_n=5):
        if self.posterior_stats is None:
            self.fit()

        df = self.posterior_stats.copy()

        # score bayésien simple :
        # rendement espéré postérieur - aversion * volatilité empirique
        df["Score"] = df["PosteriorMean"] - risk_aversion * df["SampleStd"]

        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        return df.head(top_n)

    def update_online(self, new_returns_row):
        
        if isinstance(new_returns_row, pd.Series):
            new_returns_row = new_returns_row.to_frame().T

        self.returns_df = pd.concat([self.returns_df, new_returns_row], axis=0)
        self.returns_df = self.returns_df.dropna(how="all")
        return self.fit()

    def get_posterior_distributions(self, n_samples=1000):
        
        if self.posterior_stats is None:
            self.fit()

        distributions = {}
        for _, row in self.posterior_stats.iterrows():
            samples = np.random.normal(
                loc=row["PosteriorMean"],
                scale=max(row["PosteriorStd"], 1e-8),
                size=n_samples
            )
            distributions[row["Ticker"]] = samples
        return distributions