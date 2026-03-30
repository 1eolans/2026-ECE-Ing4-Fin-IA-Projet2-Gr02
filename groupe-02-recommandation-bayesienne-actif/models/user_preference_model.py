import numpy as np


class UserPreferenceLearner:
    

    def __init__(self):
        self.lambda_value = 1.0

    def fit(self, posterior_means, sample_stds, ratings):
        posterior_means = np.asarray(posterior_means)
        sample_stds = np.asarray(sample_stds)
        ratings = np.asarray(ratings)

        candidate_lambdas = np.linspace(0.0, 5.0, 501)
        best_lambda = 1.0
        best_mse = float("inf")

        for lam in candidate_lambdas:
            preds = posterior_means - lam * sample_stds
            mse = np.mean((preds - ratings) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_lambda = lam

        self.lambda_value = float(best_lambda)
        return self.lambda_value