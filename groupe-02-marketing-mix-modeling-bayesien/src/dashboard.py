from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="MMM Bayesien - Dashboard", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


@st.cache_data
def load_csv(name: str) -> Optional[pd.DataFrame]:
    path = DOCS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def show_missing(file_name: str) -> None:
    st.warning(f"Fichier manquant: docs/{file_name}. Lance d'abord `python src/main.py`.")


st.title("Marketing Mix Modeling Bayesien")
st.caption("Visualisation des resultats multi-marche, ROI, budget et validation temporelle")

pred_df = load_csv("pred_vs_actual_hierarchical.csv")
roi_market_df = load_csv("roi_by_market_channel.csv")
roi_global_df = load_csv("roi_global_channels.csv")
budget_df = load_csv("budget_recommendation_excellent.csv")
cv_summary_df = load_csv("time_cv_summary.csv")
cv_results_df = load_csv("time_cv_results.csv")
cv_pred_df = load_csv("time_cv_predictions.csv")
posterior_df = load_csv("posterior_summary_hierarchical.csv")
betas_df = load_csv("market_channel_betas.csv")
benchmark_df = load_csv("benchmark_lightweightmmm_status.csv")
benchmark_comparison_df = load_csv("benchmark_model_comparison.csv")
benchmark_predictions_df = load_csv("benchmark_predictions.csv")

if pred_df is None:
    show_missing("pred_vs_actual_hierarchical.csv")
    st.stop()

markets = sorted(pred_df["market"].unique().tolist())
selected_market = st.sidebar.selectbox("Marche", markets)

market_data = pred_df[pred_df["market"] == selected_market].copy()
mae_market = (market_data["actual_sales"] - market_data["predicted_sales_hierarchical"]).abs().mean()
rmse_market = ((market_data["actual_sales"] - market_data["predicted_sales_hierarchical"]) ** 2).mean() ** 0.5

col1, col2, col3 = st.columns(3)
col1.metric("Marche selectionne", selected_market)
col2.metric("MAE (marche)", f"{mae_market:,.2f}")
col3.metric("RMSE (marche)", f"{rmse_market:,.2f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Model Fit", "ROI & Budget", "Validation Temporelle", "Parametres Bayesiens", "Benchmark"]
)

with tab1:
    st.subheader("Ventes reelles vs predites")
    line_df = market_data.melt(
        id_vars=["week"],
        value_vars=["actual_sales", "predicted_sales_hierarchical"],
        var_name="serie",
        value_name="sales",
    )
    fig_line = px.line(line_df, x="week", y="sales", color="serie", markers=True)
    st.plotly_chart(fig_line, width="stretch")

    st.subheader("Nuage de points: prediction vs reel")
    fig_scatter = px.scatter(
        market_data,
        x="actual_sales",
        y="predicted_sales_hierarchical",
        trendline="ols",
        title=f"{selected_market} - Qualite de fit",
    )
    st.plotly_chart(fig_scatter, width="stretch")

with tab2:
    st.subheader("ROI global par canal")
    if roi_global_df is None:
        show_missing("roi_global_channels.csv")
    else:
        fig_roi = px.bar(roi_global_df, x="channel", y="roi_proxy", color="channel")
        st.plotly_chart(fig_roi, width="stretch")
        st.dataframe(roi_global_df, width="stretch")

    st.subheader("ROI par marche et canal")
    if roi_market_df is None:
        show_missing("roi_by_market_channel.csv")
    else:
        fig_heat = px.density_heatmap(
            roi_market_df,
            x="channel",
            y="market",
            z="roi_proxy",
            histfunc="avg",
            text_auto=True,
        )
        st.plotly_chart(fig_heat, width="stretch")

    st.subheader("Recommandation budget")
    if budget_df is None:
        show_missing("budget_recommendation_excellent.csv")
    else:
        fig_budget = px.pie(
            budget_df,
            values="recommended_budget",
            names="channel",
            title="Repartition budget recommandee",
        )
        st.plotly_chart(fig_budget, width="stretch")
        st.dataframe(budget_df, width="stretch")

with tab3:
    st.subheader("Resume CV temporelle")
    if cv_summary_df is None:
        show_missing("time_cv_summary.csv")
    else:
        fig_cv = px.bar(cv_summary_df, x="model", y=["mae", "rmse", "r2"], barmode="group")
        st.plotly_chart(fig_cv, width="stretch")
        st.dataframe(cv_summary_df, width="stretch")

    st.subheader("Resultats par fold")
    if cv_results_df is None:
        show_missing("time_cv_results.csv")
    else:
        st.dataframe(cv_results_df, width="stretch")

    st.subheader("Predictions CV (echantillon)")
    if cv_pred_df is None:
        show_missing("time_cv_predictions.csv")
    else:
        st.dataframe(cv_pred_df.head(200), width="stretch")

with tab4:
    st.subheader("Resume posterior hierarchique")
    if posterior_df is None:
        show_missing("posterior_summary_hierarchical.csv")
    else:
        st.dataframe(posterior_df, width="stretch")

    st.subheader("Betas medias par marche")
    if betas_df is None:
        show_missing("market_channel_betas.csv")
    else:
        fig_betas = px.bar(
            betas_df,
            x="market",
            y="beta_mean",
            color="channel",
            barmode="group",
        )
        st.plotly_chart(fig_betas, width="stretch")
        st.dataframe(betas_df, width="stretch")

with tab5:
    st.subheader("Statut benchmark")
    if benchmark_df is None:
        show_missing("benchmark_lightweightmmm_status.csv")
    else:
        st.dataframe(benchmark_df, width="stretch")

    st.subheader("Comparaison metriques benchmark")
    if benchmark_comparison_df is None:
        show_missing("benchmark_model_comparison.csv")
    else:
        fig_bench = px.bar(
            benchmark_comparison_df,
            x="model",
            y=["mae", "rmse", "r2"],
            barmode="group",
            title="Hierarchical Bayesian MMM vs Google LightweightMMM (agrege hebdo)",
        )
        st.plotly_chart(fig_bench, width="stretch")
        st.dataframe(benchmark_comparison_df, width="stretch")

    st.subheader("Predictions benchmark (agrege hebdo)")
    if benchmark_predictions_df is None:
        show_missing("benchmark_predictions.csv")
    else:
        pred_cols = [c for c in benchmark_predictions_df.columns if c.startswith("pred_")]
        line_cols = ["actual_sales_aggregated"] + pred_cols
        bench_line_df = benchmark_predictions_df.melt(
            id_vars=["week"],
            value_vars=line_cols,
            var_name="serie",
            value_name="sales",
        )
        fig_bench_pred = px.line(
            bench_line_df,
            x="week",
            y="sales",
            color="serie",
            markers=True,
        )
        st.plotly_chart(fig_bench_pred, width="stretch")
        st.dataframe(benchmark_predictions_df, width="stretch")

st.markdown("---")
st.caption("Astuce: relance `python src/main.py` apres modification du modele pour rafraichir les visualisations.")
