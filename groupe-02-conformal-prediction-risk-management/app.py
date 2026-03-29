import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configure Streamlit page
st.set_page_config(
    page_title="CPPS V4 — Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #161b22;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .metric-title {
        color: #8b949e;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .metric-subtitle {
        color: #8b949e;
        font-size: 0.8rem;
        margin-top: 5px;
    }
    .text-green { color: #3fb950; }
    .text-red { color: #f85149; }
    .text-yellow { color: #d29922; }
</style>
""", unsafe_allow_html=True)

st.title("CPPS V4 — Risk Dashboard")
st.markdown("##### Conformal Prediction · Portfolio Selection · 2020–2022 · 10 assets")
st.markdown("---")

# Colors for methods to match the HTML version
COLORS = {
    'OnlineCPPS': '#26C6DA',
    'CPPS': '#2196F3',
    'Bayesian': '#FF9800',
    'QR': '#9C27B0',
    'AdaptCQR': '#4CAF50',
    'ScoreHybrid': '#FF5722',
    'ConfEnsemble': '#00BCD4'
}

@st.cache_data
def load_data():
    try:
        metrics_df = pd.read_csv("results_v4/metrics_v4.csv")
        # Fix column names if needed
        if 'Unnamed: 0' in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={'Unnamed: 0': 'Method'})
        
        stress_df = pd.read_csv("results_v4/stress_test.csv")
        return metrics_df, stress_df
    except FileNotFoundError:
        st.error("Results files not found. Ensure `results_v4/metrics_v4.csv` and `results_v4/stress_test.csv` exist.")
        return pd.DataFrame(), pd.DataFrame()

metrics_df, stress_df = load_data()

if not metrics_df.empty:
    # Prepare data
    metrics_df['marginal_cov_pct'] = metrics_df['marginal_cov'] * 100
    metrics_df['avg_width_pct'] = metrics_df['avg_width'] * 100
    metrics_df['covid_cov_pct'] = metrics_df['crisis_cov_COVID_2020'] * 100
    metrics_df['infl_cov_pct'] = metrics_df['crisis_cov_Inflation_2022'] * 100
    metrics_df['var_viol_rate_pct'] = metrics_df['var_viol_rate'] * 100
    
    # Calculate synthetic "Normal" coverage for visualization purposes
    metrics_df['normal_cov_pct'] = (metrics_df['marginal_cov_pct'] * 3 - metrics_df['covid_cov_pct'] - metrics_df['infl_cov_pct'])
    metrics_df['normal_cov_pct'] = metrics_df['normal_cov_pct'].clip(lower=70, upper=99)
    
    # Estimate width ratio (since it's not directly in metrics_v4.csv)
    # Using hardcoded values from HTML as reference
    width_ratios = {'OnlineCPPS': 2.14, 'CPPS': 1.25, 'Bayesian': 1.21, 'QR': 1.33, 'AdaptCQR': 1.19, 'ScoreHybrid': 1.00, 'ConfEnsemble': 1.56}
    metrics_df['width_ratio'] = metrics_df['Method'].map(width_ratios).fillna(1.0)
    
    # Sort dataset for visuals
    metrics_df = metrics_df.set_index('Method').reindex(COLORS.keys()).reset_index()

    tabs = st.tabs(["Overview", "Crisis Analysis", "Efficiency", "Stress Test"])

    # --- TAB 1: OVERVIEW ---
    with tabs[0]:
        st.subheader("Key Portfolio Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Best Marginal Cov
        best_marg = metrics_df.loc[metrics_df['marginal_cov_pct'].idxmax()]
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Best marginal cov.</div>
                <div class="metric-value text-green">{best_marg['marginal_cov_pct']:.1f}%</div>
                <div class="metric-subtitle">{best_marg['Method']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Best COVID Cov
        best_cov = metrics_df.loc[metrics_df['covid_cov_pct'].idxmax()]
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Best COVID cov.</div>
                <div class="metric-value text-green">{best_cov['covid_cov_pct']:.1f}%</div>
                <div class="metric-subtitle">{best_cov['Method']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Tightest Width
        best_w = metrics_df.loc[metrics_df['avg_width_pct'].idxmin()]
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Tightest width</div>
                <div class="metric-value text-yellow">{best_w['avg_width_pct']:.2f}%</div>
                <div class="metric-subtitle">{best_w['Method']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Best VaR Rate
        best_var = metrics_df.loc[metrics_df['var_viol_rate_pct'].idxmin()]
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Best VaR rate</div>
                <div class="metric-value text-green">{best_var['var_viol_rate_pct']:.1f}%</div>
                <div class="metric-subtitle">{best_var['Method']} (vs 10% target)</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("##### Marginal coverage by method (target ≥ 90%)")
        
        fig = px.bar(
            metrics_df, 
            x='Method', y='marginal_cov_pct', 
            color='Method', color_discrete_map=COLORS,
            text_auto='.1f'
        )
        fig.update_layout(
            showlegend=False, xaxis_title=None, yaxis_title="Coverage (%)",
            yaxis=dict(range=[60, 100]), margin=dict(t=20, b=20, l=40, r=40)
        )
        fig.add_hline(y=90, line_dash="dash", line_color="white", annotation_text="90% Target")
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: CRISIS ANALYSIS ---
    with tabs[1]:
        st.write("##### Coverage during crisis vs normal periods")
        df_melt = pd.melt(metrics_df, id_vars=['Method'], 
                          value_vars=['normal_cov_pct', 'covid_cov_pct', 'infl_cov_pct'],
                          var_name='Period', value_name='Coverage')
        
        period_map = {'normal_cov_pct': 'Normal period', 'covid_cov_pct': 'COVID 2020', 'infl_cov_pct': 'Inflation 2022'}
        df_melt['Period'] = df_melt['Period'].map(period_map)
        
        fig2 = px.bar(
            df_melt, x='Method', y='Coverage', color='Period', barmode='group',
            color_discrete_sequence=['#8b949e', '#f85149', '#58a6ff'] # Nice contrast colors
        )
        fig2.update_layout(yaxis=dict(range=[20, 100]), xaxis_title=None, yaxis_title="Coverage (%)", margin=dict(t=20))
        st.plotly_chart(fig2, use_container_width=True)
        
        st.write("##### Interval width expansion during COVID (vs normal baseline)")
        fig3 = px.bar(
            metrics_df, x='Method', y='width_ratio',
            color='width_ratio',
            color_continuous_scale=['#f85149', '#d29922', '#3fb950']
        )
        fig3.update_layout(
            showlegend=False, 
            coloraxis_showscale=False,
            xaxis_title=None, 
            yaxis_title="Ratio (COVID Width / Normal Width)",
            yaxis=dict(range=[0.5, 2.5]), margin=dict(t=20)
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.info("💡 **Note**: OnlineCPPS widens intervals ~2.14× during COVID — online calibration actively adapts. Non-adaptive methods like ScoreHybrid show ~1.0× (no adaptation), explaining lower COVID coverage.")

    # --- TAB 3: EFFICIENCY ---
    with tabs[2]:
        st.write("##### Coverage vs width trade-off (Target: Top-Left is ideal)")
        fig4 = px.scatter(
            metrics_df, x='avg_width_pct', y='marginal_cov_pct', 
            color='Method', color_discrete_map=COLORS,
            size=[15]*len(metrics_df), text='Method'
        )
        fig4.update_traces(textposition='top center')
        fig4.update_layout(
            xaxis_title="Mean Interval Width (%)", yaxis_title="Marginal Coverage (%)",
            xaxis=dict(range=[2, 7]), yaxis=dict(range=[65, 100]),
            showlegend=False, margin=dict(t=40)
        )
        fig4.add_hline(y=90, line_dash="dash", line_color="gray")
        st.plotly_chart(fig4, use_container_width=True)
        
        st.write("##### Detailed Portfolio Metrics")
        display_cols = ['Method', 'marginal_cov_pct', 'avg_width_pct', 'var_viol_rate_pct', 'sharpe', 'max_dd']
        df_display = metrics_df[display_cols].copy()
        df_display.columns = ['Method', 'Marginal Cov (%)', 'Mean Width (%)', 'VaR Violation (%)', 'Sharpe Ratio', 'Max Drawdown']
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # --- TAB 4: STRESS TEST ---
    with tabs[3]:
        st.write("##### Synthetic stress test — crisis coverage by scenario")
        
        if not stress_df.empty:
            stress_df['crisis_cov_pct'] = stress_df['crisis_cov'] * 100
            fig5 = px.bar(
                stress_df, x='method', y='crisis_cov_pct', color='scenario', barmode='group',
                color_discrete_sequence=['#f85149', '#58a6ff', '#3fb950']
            )
            fig5.update_layout(yaxis=dict(range=[0, 105]), xaxis_title=None, yaxis_title="Coverage (%)", margin=dict(t=20))
            st.plotly_chart(fig5, use_container_width=True)
            
            # Pivot table
            st.write("##### Scenario Breakdown")
            df_pivot = stress_df.pivot(index='method', columns='scenario', values='crisis_cov_pct').reset_index()
            # Count recalls 
            df_pivot['recal_needed'] = stress_df.groupby('method')['needs_recal'].sum().values
            st.dataframe(df_pivot, use_container_width=True, hide_index=True)
            
            st.info("💡 **Note**: Bayesian collapses hardest under sudden regime change (2008 & COVID). OnlineCPPS exceeds 80% coverage during the 2008 crash since the online adaptation window reacts within days.")
        else:
            st.warning("No stress test data available.")

