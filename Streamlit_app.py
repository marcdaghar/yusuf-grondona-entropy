#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Interactive Dashboard – Yusuf Counter-Cycle Model

Run with: streamlit run streamlit_app.py

Author: Marc Daghar
Licence: CC BY-SA 4.0
Mention: Free Dr Aafia Siddiqui !
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from yusuf_model import YusufConfig, YusufSystem, CapitalistSystem, ScenarioComparator
from grondona_crd import Commodity, CommodityReserveDepartment, GrondonaSimulator
from statistical_validation import StatisticalValidator


st.set_page_config(
    page_title="Yusuf Counter-Cycle – Monetary Model",
    page_icon="🪙",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a472a, #2d6a4f);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .quran-quote {
        font-style: italic;
        text-align: center;
        padding: 1rem;
        background: #f0f2f6;
        border-radius: 10px;
        margin: 1rem 0;
        border-right: 4px solid #2d6a4f;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🪙 YUSUF COUNTER-CYCLE MODEL</h1>
    <p>From Usury to Resilience – Bimetallic Alternative to Debt-Based Money</p>
    <p style="font-size: 0.8rem;">Based on Surah Yusuf (12:47-48) | Grondona System | Neurocognitive Agents</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="quran-quote">
    <strong>Surah Yusuf (12:47-48)</strong><br>
    "For seven years you shall sow as usual. What you reap, leave it in its ears, except a little that you eat.
    Then after that will come seven hard years which will devour what you have prepared for them..."
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Parameters")
    
    st.subheader("📐 Time")
    T = st.slider("Years", 20, 200, 100, 10)
    dt = st.slider("Time step (years)", 0.05, 0.5, 0.1, 0.05)
    
    st.subheader("📊 Economy")
    need = st.slider("Minimum consumption need", 0.3, 1.0, 0.7, 0.05)
    P_amplitude = st.slider("Production amplitude", 0.2, 0.8, 0.5, 0.05)
    period = st.slider("Cycle period (years)", 8, 20, 14, 1)
    interest_rate = st.slider("Interest rate (capitalist)", 0.0, 0.15, 0.05, 0.01)
    
    st.subheader("🎮 Gamification")
    gamification = st.checkbox("Enable social credit", value=True)
    compliance_threshold = st.slider("Compliance threshold", 0.5, 0.95, 0.8, 0.05)
    
    st.subheader("🌊 Noise")
    noise = st.slider("Production noise", 0.0, 0.1, 0.03, 0.01)
    
    run_button = st.button("▶ Run Simulation", use_container_width=True)


config = YusufConfig(
    T=T, dt=dt, need=need, P_amplitude=P_amplitude, period=period,
    interest_rate=interest_rate, gamification_enabled=gamification,
    compliance_threshold=compliance_threshold, noise_amplitude=noise
)


@st.cache_data
def run_simulation(config):
    comparator = ScenarioComparator(config)
    return comparator.run_single()


if run_button:
    with st.spinner("Running simulation..."):
        y_res, c_res = run_simulation(config)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Yusuf Final Stock", f"{y_res.final_stock:.2f}",
                  delta=f"{y_res.final_stock - c_res.final_stock:+.2f} vs Capitalist")
    with col2:
        st.metric("Capitalist Final Stock", f"{c_res.final_stock:.2f}")
    with col3:
        delta_solv = y_res.solvency_rate - c_res.solvency_rate
        st.metric("Solvency", f"{y_res.solvency_rate:.1f}%",
                  delta=f"{delta_solv:+.1f}% for Yusuf")
    with col4:
        delta_vol = (1 - y_res.consumption_volatility / max(c_res.consumption_volatility, 0.01)) * 100
        st.metric("Stability", f"Vol={y_res.consumption_volatility:.3f}",
                  delta=f"{delta_vol:+.0f}% more stable")
    
    # Interactive chart
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Production Cycle", "Stock Evolution",
                                        "Consumption", "Coverage Ratio"))
    
    fig.add_trace(go.Scatter(x=y_res.t, y=y_res.P, name="Production", line=dict(color='black')),
                  row=1, col=1)
    fig.add_hline(y=config.P_bar, line_dash="dash", line_color="green",
                  annotation_text="Abundance", row=1, col=1)
    fig.add_hline(y=config.P_underline, line_dash="dash", line_color="red",
                  annotation_text="Scarcity", row=1, col=1)
    
    fig.add_trace(go.Scatter(x=c_res.t, y=c_res.S, name="Capitalist", line=dict(color='red')),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=y_res.t, y=y_res.S, name="Yusuf", line=dict(color='blue')),
                  row=1, col=2)
    
    fig.add_trace(go.Scatter(x=c_res.t, y=c_res.C, name="Capitalist", line=dict(color='red', opacity=0.7)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=y_res.t, y=y_res.C, name="Yusuf", line=dict(color='blue', opacity=0.7)),
                  row=2, col=1)
    fig.add_hline(y=config.need, line_dash="dot", line_color="black", row=2, col=1)
    
    fig.add_trace(go.Scatter(x=c_res.t, y=c_res.coverage_ratio, name="Capitalist", line=dict(color='red')),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=y_res.t, y=y_res.coverage_ratio, name="Yusuf", line=dict(color='blue')),
                  row=2, col=2)
    fig.add_hline(y=1, line_dash="dash", line_color="black", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    with st.expander("📊 Detailed Results"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Yusuf System")
            st.dataframe(pd.DataFrame({
                "Metric": ["Final stock", "Mean consumption", "Volatility", "Solvency"],
                "Value": [f"{y_res.final_stock:.2f}", f"{y_res.mean_consumption:.2f}",
                          f"{y_res.consumption_volatility:.4f}", f"{y_res.solvency_rate:.1f}%"]
            }), hide_index=True)
        with col2:
            st.subheader("Capitalist System")
            st.dataframe(pd.DataFrame({
                "Metric": ["Final stock", "Mean consumption", "Volatility", "Solvency"],
                "Value": [f"{c_res.final_stock:.2f}", f"{c_res.mean_consumption:.2f}",
                          f"{c_res.consumption_volatility:.4f}", f"{c_res.solvency_rate:.1f}%"]
            }), hide_index=True)
    
    # Interpretation
    st.markdown("""
    <div class="quran-quote">
        <strong>💡 Interpretation</strong><br>
        The Yusuf counter-cycle system demonstrates higher solvency, lower volatility,
        and greater resilience to shocks than the interest-based capitalist system.
        This validates the principle of saving in abundance and consuming from stock in scarcity.
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👈 Configure parameters in the sidebar and click 'Run Simulation'")


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>🪙 Yusuf Counter-Cycle Model – CC BY-SA 4.0 | Marc Daghar | Free Dr Aafia Siddiqui !</p>
    <p>🌿 Blessed are the cracked, for they shall let in the light.</p>
</div>
""", unsafe_allow_html=True)
