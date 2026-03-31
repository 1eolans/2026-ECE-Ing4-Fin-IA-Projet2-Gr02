from pathlib import Path
import argparse

import pandas as pd

from benchmark import run_lightweightmmm_benchmark
from cv import run_temporal_cv_multimarket
from data import generate_synthetic_multimarket_mmm_data
from features import build_multimarket_mmm_features
from model import (
    fit_hierarchical_bayesian_mmm,
    hierarchical_posterior_means,
    hierarchical_posterior_summary,
    predict_hierarchical_sales,
)
from optimize import estimate_channel_roi_multimarket, recommend_budget


def run_pipeline_excellent(
    output_dir: Path,
    n_markets: int = 4,
    n_weeks: int = 130,
    main_draws: int = 450,
    main_tune: int = 450,
    cv_draws: int = 180,
    cv_tune: int = 180,
) -> None:
    channels = ["tv", "facebook", "google"]

    adstock_alpha = {"tv": 0.50, "facebook": 0.35, "google": 0.45}
    hill_ec50 = {"tv": 220.0, "facebook": 90.0, "google": 120.0}
    hill_slope = {"tv": 1.7, "facebook": 2.1, "google": 2.0}

    raw_df = generate_synthetic_multimarket_mmm_data(n_markets=n_markets, n_weeks=n_weeks, seed=42)

    features_df = build_multimarket_mmm_features(
        raw_df,
        channels=channels,
        adstock_alpha=adstock_alpha,
        hill_ec50=hill_ec50,
        hill_slope=hill_slope,
        market_col="market",
        week_col="week",
    )

    market_names = sorted(raw_df["market"].unique().tolist())
    market_to_idx = {m: i for i, m in enumerate(market_names)}
    market_idx = raw_df["market"].map(market_to_idx).to_numpy(dtype=int)

    _, idata = fit_hierarchical_bayesian_mmm(
        features=features_df,
        sales=raw_df["sales"],
        market_idx=market_idx,
        n_markets=len(market_names),
        draws=main_draws,
        tune=main_tune,
        chains=2,
        random_seed=42,
    )

    summary_df = hierarchical_posterior_summary(idata)
    market_beta_df, market_intercept_map = hierarchical_posterior_means(
        idata,
        channels=channels,
        market_names=market_names,
    )

    pred = predict_hierarchical_sales(
        features=features_df,
        market_series=raw_df["market"],
        market_beta_df=market_beta_df,
        market_intercept_map=market_intercept_map,
    )

    pred_df = pd.DataFrame(
        {
            "market": raw_df["market"],
            "week": raw_df["week"],
            "actual_sales": raw_df["sales"],
            "predicted_sales_hierarchical": pred,
        }
    )

    roi_by_market_df, roi_global_df = estimate_channel_roi_multimarket(
        raw_df=raw_df,
        transformed_features=features_df,
        market_beta_df=market_beta_df,
        channels=channels,
        market_col="market",
    )

    budget_df = recommend_budget(roi_global_df, total_budget=120_000.0, min_share=0.10)

    cv_df, cv_pred_df = run_temporal_cv_multimarket(
        raw_df=raw_df,
        channels=channels,
        adstock_alpha=adstock_alpha,
        hill_ec50=hill_ec50,
        hill_slope=hill_slope,
        n_folds=3,
        horizon=8,
        min_train_weeks=70,
        draws=cv_draws,
        tune=cv_tune,
    )

    cv_summary = (
        cv_df.groupby("model", as_index=False)[["mae", "rmse", "r2"]]
        .mean()
        .sort_values("rmse")
    )

    benchmark_status_df, benchmark_comparison_df, benchmark_predictions_df = run_lightweightmmm_benchmark(
        raw_df=raw_df,
        channels=channels,
        bayesian_predictions=pred,
        market_col="market",
        week_col="week",
        target_col="sales",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "posterior_summary_hierarchical.csv", index=False)
    market_beta_df.to_csv(output_dir / "market_channel_betas.csv", index=False)
    pred_df.to_csv(output_dir / "pred_vs_actual_hierarchical.csv", index=False)
    roi_by_market_df.to_csv(output_dir / "roi_by_market_channel.csv", index=False)
    roi_global_df.to_csv(output_dir / "roi_global_channels.csv", index=False)
    budget_df.to_csv(output_dir / "budget_recommendation_excellent.csv", index=False)
    cv_df.to_csv(output_dir / "time_cv_results.csv", index=False)
    cv_summary.to_csv(output_dir / "time_cv_summary.csv", index=False)
    cv_pred_df.to_csv(output_dir / "time_cv_predictions.csv", index=False)
    benchmark_status_df.to_csv(output_dir / "benchmark_lightweightmmm_status.csv", index=False)
    benchmark_comparison_df.to_csv(output_dir / "benchmark_model_comparison.csv", index=False)
    benchmark_predictions_df.to_csv(output_dir / "benchmark_predictions.csv", index=False)

    print("Pipeline EXCELLENT termine. Fichiers generes dans:", output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMM excellent pipeline.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for generated CSV files (default: <project>/docs).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster smoke configuration (fewer samples and shorter horizon).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    project_root = Path(__file__).resolve().parent.parent
    docs_dir = Path(args.output_dir) if args.output_dir else (project_root / "docs")

    if args.quick:
        run_pipeline_excellent(
            output_dir=docs_dir,
            n_markets=3,
            n_weeks=90,
            main_draws=120,
            main_tune=120,
            cv_draws=80,
            cv_tune=80,
        )
    else:
        run_pipeline_excellent(output_dir=docs_dir)
