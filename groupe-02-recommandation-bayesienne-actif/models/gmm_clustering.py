import pandas as pd
from sklearn.mixture import GaussianMixture


class AssetClusterer:
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = GaussianMixture(n_components=n_clusters, random_state=random_state)

    def fit_predict(self, returns_df):
        features = pd.DataFrame({
            "Mean": returns_df.mean(),
            "Std": returns_df.std(),
        })

        labels = self.model.fit_predict(features.values)
        features["Cluster"] = labels
        features["Ticker"] = features.index
        return features.reset_index(drop=True)