import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import build_multimarket_mmm_features
from model import (
    fit_hierarchical_bayesian_mmm,
    hierarchical_posterior_means,
    predict_hierarchical_sales,
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _fit_ridge_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, channels: list[str]) -> np.ndarray:
    x_train = train_df[channels].copy()
    x_test = test_df[channels].copy()

    # Add market fixed effects for a fair comparison with hierarchical model.
    train_dummies = pd.get_dummies(train_df["market"], prefix="market")
    test_dummies = pd.get_dummies(test_df["market"], prefix="market")

    all_cols = sorted(set(train_dummies.columns).union(set(test_dummies.columns)))
    train_dummies = train_dummies.reindex(columns=all_cols, fill_value=0)
    test_dummies = test_dummies.reindex(columns=all_cols, fill_value=0)

    x_train = pd.concat([x_train.reset_index(drop=True), train_dummies.reset_index(drop=True)], axis=1)
    x_test = pd.concat([x_test.reset_index(drop=True), test_dummies.reset_index(drop=True)], axis=1)

    y_train = train_df["sales"].to_numpy(dtype=float)

    model = Ridge(alpha=1.0)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def run_temporal_cv_multimarket(
    raw_df: pd.DataFrame,
    channels: list[str],
    adstock_alpha: dict[str, float],
    hill_ec50: dict[str, float],
    hill_slope: dict[str, float],
    n_folds: int = 3,
    horizon: int = 8,
    min_train_weeks: int = 70,
    draws: int = 220,
    tune: int = 220,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window temporal CV with hierarchical Bayesian MMM vs Ridge baseline."""
    max_week = int(raw_df["week"].max())
    cutoffs = []
    current = min_train_weeks
    while current + horizon <= max_week and len(cutoffs) < n_folds:
        cutoffs.append(current)
        current += horizon

    rows = []
    fold_predictions = []

    market_names = sorted(raw_df["market"].unique().tolist())
    market_to_idx = {m: i for i, m in enumerate(market_names)}

    for fold_id, cutoff in enumerate(cutoffs, start=1):
        train_mask = raw_df["week"] <= cutoff
        test_mask = (raw_df["week"] > cutoff) & (raw_df["week"] <= cutoff + horizon)

        train_df = raw_df.loc[train_mask].copy()
        test_df = raw_df.loc[test_mask].copy()

        features_full = build_multimarket_mmm_features(
            raw_df,
            channels=channels,
            adstock_alpha=adstock_alpha,
            hill_ec50=hill_ec50,
            hill_slope=hill_slope,
            market_col="market",
            week_col="week",
        )

        x_train = features_full.loc[train_df.index]
        x_test = features_full.loc[test_df.index]

        train_market_idx = train_df["market"].map(market_to_idx).to_numpy(dtype=int)

        _, idata = fit_hierarchical_bayesian_mmm(
            features=x_train,
            sales=train_df["sales"],
            market_idx=train_market_idx,
            n_markets=len(market_names),
            draws=draws,
            tune=tune,
            chains=2,
            random_seed=42 + fold_id,
        )

        market_beta_df, market_intercept_map = hierarchical_posterior_means(
            idata,
            channels=channels,
            market_names=market_names,
        )

        bayes_pred = predict_hierarchical_sales(
            features=x_test,
            market_series=test_df["market"],
            market_beta_df=market_beta_df,
            market_intercept_map=market_intercept_map,
        )

        ridge_pred = _fit_ridge_baseline(train_df=x_train.join(train_df[["market", "sales"]]), test_df=x_test.join(test_df[["market", "sales"]]), channels=channels)

        y_true = test_df["sales"].to_numpy(dtype=float)
        bayes_metrics = _metrics(y_true, bayes_pred)
        ridge_metrics = _metrics(y_true, ridge_pred)

        rows.append(
            {
                "fold": fold_id,
                "cutoff_week": cutoff,
                "model": "hierarchical_bayesian_mmm",
                **bayes_metrics,
            }
        )
        rows.append(
            {
                "fold": fold_id,
                "cutoff_week": cutoff,
                "model": "ridge_market_fixed_effects",
                **ridge_metrics,
            }
        )

        fold_pred_df = pd.DataFrame(
            {
                "fold": fold_id,
                "market": test_df["market"].to_numpy(),
                "week": test_df["week"].to_numpy(),
                "actual_sales": y_true,
                "pred_hierarchical_bayesian_mmm": bayes_pred,
                "pred_ridge_market_fixed_effects": ridge_pred,
            }
        )
        fold_predictions.append(fold_pred_df)

    return pd.DataFrame(rows), pd.concat(fold_predictions, ignore_index=True)
