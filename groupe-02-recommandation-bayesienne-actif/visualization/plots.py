import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_posterior_distributions(distributions):
    n_assets = len(distributions)
    fig, axes = plt.subplots(n_assets, 1, figsize=(10, 3 * n_assets), squeeze=False)

    for i, (ticker, samples) in enumerate(distributions.items()):
        ax = axes[i, 0]
        sns.histplot(samples, kde=True, ax=ax, bins=30)
        ax.set_title(f"Distribution postérieure de la moyenne - {ticker}")
        ax.set_xlabel("Rendement moyen")
        ax.set_ylabel("Fréquence")

    plt.tight_layout()
    return fig


def plot_cumulative_returns(backtest_df):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(backtest_df["Date"], backtest_df["CumulativeReturn"], linewidth=2)
    ax.set_title("Backtesting - Rendement cumulé")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement cumulé")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_risk_return_scatter(returns_df, recommendations=None):
    stats = pd.DataFrame({
        "Ticker": returns_df.columns,
        "Mean": returns_df.mean().values,
        "Std": returns_df.std().values
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(stats["Std"], stats["Mean"], s=100)

    for _, row in stats.iterrows():
        ax.annotate(row["Ticker"], (row["Std"], row["Mean"]), xytext=(5, 5), textcoords="offset points")

    if recommendations is not None and not recommendations.empty:
        selected = stats[stats["Ticker"].isin(recommendations["Ticker"])]
        ax.scatter(selected["Std"], selected["Mean"], s=200, marker="*", label="Recommandés")

    ax.set_title("Risque / Rendement")
    ax.set_xlabel("Volatilité")
    ax.set_ylabel("Rendement moyen")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig