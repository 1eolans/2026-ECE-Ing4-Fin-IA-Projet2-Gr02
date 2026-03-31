import numpy as np
import pandas as pd
import inspect
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _patch_where_keyword_compat() -> None:
    """
    Compatibility patch for libs that call where(condition=..., x=..., y=...)
    while newer signatures enforce positional-only arguments.
    """
    # Patch numpy.where
    np_where_orig = np.where

    def _np_where_compat(*args, **kwargs):
        if kwargs and not args and {"condition", "x", "y"}.issubset(kwargs.keys()):
            return np_where_orig(kwargs["condition"], kwargs["x"], kwargs["y"])
        return np_where_orig(*args, **kwargs)

    np.where = _np_where_compat  # type: ignore[assignment]

    # Patch jax.numpy.where when available
    try:
        import jax.numpy as jnp

        jnp_where_orig = jnp.where

        def _jnp_where_compat(*args, **kwargs):
            if kwargs and not args and {"condition", "x", "y"}.issubset(kwargs.keys()):
                return jnp_where_orig(kwargs["condition"], kwargs["x"], kwargs["y"])
            return jnp_where_orig(*args, **kwargs)

        jnp.where = _jnp_where_compat  # type: ignore[assignment]
    except Exception:
        pass


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def _as_1d_prediction(pred: np.ndarray, expected_len: int) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        if arr.shape[1] == expected_len:
            return arr.mean(axis=0).astype(float)
        if arr.shape[0] == expected_len:
            return arr.mean(axis=1).astype(float)
        return arr.reshape(-1)[:expected_len].astype(float)

    # Generic fallback: average all sample dimensions and keep last dimension as time.
    reduce_axes = tuple(range(arr.ndim - 1))
    reduced = arr.mean(axis=reduce_axes)
    return np.asarray(reduced).reshape(-1)[:expected_len].astype(float)


def run_lightweightmmm_benchmark(
    raw_df: pd.DataFrame,
    channels: list[str],
    bayesian_predictions: np.ndarray,
    market_col: str = "market",
    week_col: str = "week",
    target_col: str = "sales",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare hierarchical Bayesian MMM vs Google LightweightMMM on aggregated weekly data.

    Returns:
    - status_df: install/run status and diagnostics
    - comparison_df: MAE/RMSE/R2 by model
    - predictions_df: weekly actual vs predicted series
    """
    work_df = raw_df.copy()
    work_df["pred_bayesian_hierarchical"] = np.asarray(bayesian_predictions, dtype=float)

    agg_map = {target_col: "sum", "pred_bayesian_hierarchical": "sum"}
    for ch in channels:
        agg_map[f"spend_{ch}"] = "sum"

    weekly_df = (
        work_df.groupby(week_col, as_index=False)
        .agg(agg_map)
        .sort_values(week_col)
        .reset_index(drop=True)
    )

    y_true = weekly_df[target_col].to_numpy(dtype=float)
    y_bayes = weekly_df["pred_bayesian_hierarchical"].to_numpy(dtype=float)
    metrics_bayes = _metrics(y_true, y_bayes)

    comparison_rows = [
        {
            "model": "hierarchical_bayesian_mmm_aggregated",
            **metrics_bayes,
        }
    ]

    predictions_df = pd.DataFrame(
        {
            "week": weekly_df[week_col].to_numpy(),
            "actual_sales_aggregated": y_true,
            "pred_hierarchical_bayesian_aggregated": y_bayes,
        }
    )

    status = {
        "benchmark": "google_lightweightmmm",
        "status": "not_installed",
        "details": "lightweight_mmm package is not installed.",
    }

    try:
        _patch_where_keyword_compat()
        from lightweight_mmm.lightweight_mmm import LightweightMMM

        media = weekly_df[[f"spend_{ch}" for ch in channels]].to_numpy(dtype=float)
        # Media prior by channel: normalized mean spend share.
        media_prior = media.mean(axis=0)
        media_prior = media_prior / np.maximum(media_prior.sum(), 1e-9)

        model = LightweightMMM(model_name="adstock")
        fit_kwargs = {
            "media": media,
            "target": y_true,
            "number_warmup": 300,
            "number_samples": 300,
            "number_chains": 1,
            "seed": 42,
        }
        fit_sig = inspect.signature(model.fit)
        if "media_prior" in fit_sig.parameters:
            fit_kwargs["media_prior"] = media_prior

        model.fit(**fit_kwargs)

        pred_kwargs = {"media": media}
        pred_sig = inspect.signature(model.predict)
        if "media_prior" in pred_sig.parameters:
            pred_kwargs["media_prior"] = media_prior
        pred_lmmm = model.predict(**pred_kwargs)
        y_lmmm = _as_1d_prediction(pred_lmmm, expected_len=len(y_true))
        metrics_lmmm = _metrics(y_true, y_lmmm)

        comparison_rows.append(
            {
                "model": "google_lightweightmmm_aggregated",
                **metrics_lmmm,
            }
        )
        predictions_df["pred_google_lightweightmmm_aggregated"] = y_lmmm

        status = {
            "benchmark": "google_lightweightmmm",
            "status": "available_and_ran",
            "details": "LightweightMMM installed and benchmark executed successfully.",
        }
    except ModuleNotFoundError:
        status = {
            "benchmark": "google_lightweightmmm",
            "status": "not_installed",
            "details": "lightweight_mmm package is not installed in the active Python environment.",
        }
    except Exception as exc:
        status = {
            "benchmark": "google_lightweightmmm",
            "status": "installed_but_failed",
            "details": f"Benchmark failed at runtime: {type(exc).__name__}: {exc}",
        }

    status_df = pd.DataFrame([status])
    comparison_df = pd.DataFrame(comparison_rows).sort_values("rmse").reset_index(drop=True)
    return status_df, comparison_df, predictions_df
