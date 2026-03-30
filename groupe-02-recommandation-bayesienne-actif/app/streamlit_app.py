import streamlit as st
import pandas as pd

from services.data_loader import load_returns
from services.risk_metrics import summarize_asset_risk
from models.bayesian_recommender import BayesianRecommender
from models.gmm_clustering import AssetClusterer
from services.backtester import Backtester
from visualization.plots import (
    plot_posterior_distributions,
    plot_cumulative_returns,
    plot_risk_return_scatter,
)


st.set_page_config(page_title="Recommandation Bayésienne d'Actifs", layout="wide")

st.title("📈 Recommandation Bayésienne d'Actifs Financiers")
st.write("Système léger de recommandation d'actifs basé sur une estimation bayésienne des rendements.")

tickers_input = st.text_input(
    "Tickers (séparés par des virgules)",
    "AAPL,MSFT,GOOGL,AMZN,TSLA,JPM,NVDA,META"
)

period = st.selectbox("Période historique", ["1y", "2y", "3y", "5y"], index=2)
risk_aversion = st.slider("Aversion au risque (λ)", 0.0, 5.0, 1.0, 0.1)
top_n = st.slider("Nombre d'actifs recommandés", 1, 8, 3)

run_button = st.button("Lancer l'analyse")

if run_button:
    try:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        prices, returns = load_returns(tickers, period=period)

        st.subheader("Aperçu des rendements")
        st.dataframe(returns.tail())

        recommender = BayesianRecommender(returns)
        posterior_stats = recommender.fit()
        recommendations = recommender.recommend(risk_aversion=risk_aversion, top_n=top_n)

        st.subheader("📌 Statistiques postérieures bayésiennes")
        st.dataframe(posterior_stats)

        st.subheader("🏆 Recommandations")
        st.dataframe(recommendations)

        st.subheader("📉 Analyse risque / rendement")
        fig_rr = plot_risk_return_scatter(returns, recommendations)
        st.pyplot(fig_rr)

        st.subheader("⚠️ Mesures de risque")
        risk_summary = summarize_asset_risk(returns, alpha=0.05)
        st.dataframe(pd.DataFrame(risk_summary))

        st.subheader("🧠 Clustering des actifs")
        clusterer = AssetClusterer(n_clusters=3)
        cluster_df = clusterer.fit_predict(returns)
        st.dataframe(cluster_df)

        st.subheader("📊 Distributions postérieures")
        distributions = recommender.get_posterior_distributions(n_samples=1000)
        fig_post = plot_posterior_distributions(distributions)
        st.pyplot(fig_post)

        st.subheader("🔁 Backtesting")
        if len(returns) > 260:
            backtester = Backtester(
                returns_df=returns,
                window=252,
                top_n=min(top_n, len(returns.columns)),
                risk_aversion=risk_aversion
            )
            backtest_df = backtester.run()
            st.dataframe(backtest_df.tail())

            fig_bt = plot_cumulative_returns(backtest_df)
            st.pyplot(fig_bt)
        else:
            st.info("Pas assez de données pour un backtesting avec une fenêtre de 252 jours.")

    except Exception as e:
        st.error(f"Erreur : {e}")