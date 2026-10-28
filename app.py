"""
QuantEdge AI — Institutional Indian Stock Market (NSE / BSE) Quantitative RL Terminal
High-performance Streamlit Web Application for Reinforcement Learning Trading,
Actor-Critic Neural Inference, Multi-Agent Consensus, and Institutional Risk Governance.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data import (
    load_ticker_data,
    download_ticker_to_csv,
    normalize_ticker,
    DEFAULT_UNIVERSE,
    MACRO_TICKERS,
    get_macro_context,
    DATA_DIR,
)
from core.features import compute_features
from core.models import (
    RLForecastingSystem,
    prepare_market_state_matrix,
    ACTION_NAMES,
    ACTION_WEIGHTS,
    ActorCriticNetwork,
)
from core.regime import MacroRegimeClassifier
from core.risk import RiskGovernor
from core.sentiment import MarketSentimentEngine
from core.explainability import PolicyExplainer, FEATURE_NAMES
from core.agents import MultiAgentConsensusEngine
from core.portfolio import InstitutionalPortfolioEngine

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Modern Institutional CSS Design
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantEdge AI — NSE/BSE Quantitative RL Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Institutional CSS
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

code, pre, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Background & Surface Colors */
.stApp {
    background-color: #080C14;
    color: #E2E8F0;
}

/* Custom Card Container */
.quant-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.75) 100%);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.45);
    backdrop-filter: blur(12px);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.quant-card:hover {
    border-color: rgba(96, 165, 250, 0.45);
}

.quant-card-glass {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 16px;
    backdrop-filter: blur(8px);
}

/* Signal Badges */
.badge-strong-buy {
    background: linear-gradient(135deg, #059669 0%, #10B981 100%);
    color: #FFFFFF;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    display: inline-block;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.4);
}

.badge-buy {
    background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%);
    color: #FFFFFF;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    display: inline-block;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.35);
}

.badge-hold {
    background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
    color: #0F172A;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    display: inline-block;
    box-shadow: 0 0 15px rgba(245, 158, 11, 0.35);
}

.badge-sell {
    background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
    color: #FFFFFF;
    padding: 6px 14px;
    border-radius: 8px;
    font-weight: 800;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    display: inline-block;
    box-shadow: 0 0 15px rgba(239, 68, 68, 0.4);
}

/* Metric Display Unit */
.metric-tile {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: left;
}

.metric-label {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}

.metric-val {
    font-size: 1.45rem;
    font-weight: 700;
    color: #F8FAFC;
    font-family: 'JetBrains Mono', monospace;
}

.metric-sub {
    font-size: 0.75rem;
    margin-top: 2px;
}

/* Accent Gradients */
.text-gradient {
    background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background-color: #0B111E !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Caching & Optimization Helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def get_cached_ticker_data(symbol: str):
    """Loads ticker data from local CSV or downloads if missing."""
    clean_sym = normalize_ticker(symbol)
    df = load_ticker_data(clean_sym, period="1y", interval="1d")
    feat_df = compute_features(df)
    return feat_df


@st.cache_data(ttl=900)
def get_cached_macro():
    """Fetches macro context (^INDIAVIX, ^NSEI)."""
    return get_macro_context()


@st.cache_resource
def get_trained_rl_system(symbol: str, epochs: int = 15, vix_val: float = 14.5, nifty_ret: float = 0.001):
    """Trains and caches the RL forecasting system for a given symbol."""
    feat_df = get_cached_ticker_data(symbol)
    rl_sys = RLForecastingSystem()
    train_stats = rl_sys.train_on_dataframe(
        feat_df,
        vix_val=vix_val,
        nifty_ret=nifty_ret,
        epochs=epochs,
        repeat_cycles=2,
    )
    eval_res = rl_sys.evaluate_strategy(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
    curr_pred = rl_sys.predict_current_signal(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
    return rl_sys, train_stats, eval_res, curr_pred


# -----------------------------------------------------------------------------
# Sidebar Navigation & Global Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <h2 style="margin: 0; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;">
            <span class="text-gradient">QuantEdge</span> AI
        </h2>
        <p style="margin: 2px 0 0 0; font-size: 0.75rem; color: #94A3B8; letter-spacing: 0.05em;">
            INSTITUTIONAL QUANT RL TERMINAL • NSE / BSE
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Real-time Macro Ticker Bar in Sidebar
    try:
        vix_val, nifty_ret = get_cached_macro()
    except Exception:
        vix_val, nifty_ret = 14.2, 0.002

    regime_info = MacroRegimeClassifier.classify_regime(vix_val, nifty_ret)

    reg_color = "#10B981" if not regime_info["is_crisis"] and regime_info["risk_level"] == "LOW" else ("#F59E0B" if regime_info["risk_level"] == "MEDIUM" else "#EF4444")

    st.markdown(f"""
    <div class="quant-card-glass" style="margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">INDIA MACRO REGIME</span>
            <span style="font-size: 0.7rem; background: rgba(59,130,246,0.15); color: {reg_color}; padding: 2px 8px; border-radius: 4px; font-weight: 700;">{regime_info['regime']}</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
            <div>
                <div style="font-size: 0.7rem; color: #64748B;">INDIA VIX</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">{vix_val:.2f}</div>
            </div>
            <div>
                <div style="font-size: 0.7rem; color: #64748B;">NIFTY 1D</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {'#10B981' if nifty_ret >= 0 else '#EF4444'};">{nifty_ret*100:+.2f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Engine Parameters")
    selected_symbol = st.selectbox(
        "Select NSE Asset",
        options=DEFAULT_UNIVERSE,
        index=0,
        help="Select any large-cap NSE bluechip from the institutional trading universe."
    )

    ac_epochs = st.slider(
        "Actor-Critic Rollout Epochs",
        min_value=1,
        max_value=30,
        value=15,
        step=1,
        help="Training iterations per Actor-Critic policy gradient update."
    )

    capital_book_crores = st.number_input(
        "Institutional Book Capital (₹ Crores)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.5,
        help="Institutional capital base in INR Crores for risk sizing."
    )
    capital_book_inr = capital_book_crores * 10_000_000.0

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.72rem; color: #64748B; line-height: 1.4;">
        <b>SEBI Algo Compliance:</b><br>
        • Half-Kelly Position Bounding<br>
        • Asymmetric Semi-Variance Penalty<br>
        • Exact 9,573-Param Actor-Critic Network<br>
        • 95% Daily Historical & Parametric VaR
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Application Header
# -----------------------------------------------------------------------------
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 16px; margin-bottom: 24px;">
    <div>
        <h1 style="font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: -0.03em;">
            <span class="text-gradient">QuantEdge AI</span> <span style="font-size: 1.2rem; color: #64748B; font-weight: 400;">| Institutional RL Engine</span>
        </h1>
        <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
            Deep Actor-Critic Quantitative Execution, Multi-Agent Consensus & SEBI Risk Governance
        </p>
    </div>
    <div style="text-align: right;">
        <span style="font-size: 0.75rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Execution Mode</span><br>
        <span style="font-size: 0.85rem; color: #10B981; font-weight: 700; background: rgba(16,185,129,0.1); padding: 4px 10px; border-radius: 6px; border: 1px solid rgba(16,185,129,0.25);">
            ● LIVE DETERMINISTIC INFERENCE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Top-Level Tab Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Single-Stock RL Terminal",
    "10-Stock Portfolio (INR 1 Cr)",
    "Deep Explainability Lab",
    "Actor-Critic Training Sandbox",
    "Macro Regimes & Sentiment",
])


# =============================================================================
# TAB 1: Single-Stock RL Trading Terminal
# =============================================================================
with tab1:
    with st.spinner(f"Ingesting local CSV & executing Actor-Critic for {selected_symbol}..."):
        feat_df = get_cached_ticker_data(selected_symbol)
        rl_sys, train_stats, eval_res, raw_pred = get_trained_rl_system(
            selected_symbol, epochs=ac_epochs, vix_val=vix_val, nifty_ret=nifty_ret
        )

        curr_price = float(feat_df["Close"].iloc[-1])
        prev_price = float(feat_df["Close"].iloc[-2])
        price_chg = curr_price - prev_price
        price_chg_pct = (price_chg / prev_price) * 100.0
        atr_val = float(feat_df["ATRr_14"].iloc[-1])

        # Run Multi-Agent Consensus
        consensus_engine = MultiAgentConsensusEngine()
        deliberation = consensus_engine.deliberate(
            symbol=selected_symbol,
            rl_prediction=raw_pred,
            vix_val=vix_val,
            nifty_ret=nifty_ret,
            current_price=curr_price,
            atr_14=atr_val,
        )

        final_signal = deliberation["final_signal"]
        alloc_fraction = deliberation["allocation_fraction"]
        conviction = deliberation["confidence"]

        # VaR on single stock
        single_asset_capital = capital_book_inr * min(0.20, alloc_fraction)
        var_res = RiskGovernor.compute_daily_var_95(feat_df["Ret_1d"].values, portfolio_capital_inr=max(100_000, single_asset_capital))

    # Signal Badge Helper
    def get_badge_html(sig):
        if sig == "STRONG_BUY":
            return '<span class="badge-strong-buy">STRONG BUY</span>'
        elif sig == "BUY":
            return '<span class="badge-buy">BUY</span>'
        elif sig == "HOLD":
            return '<span class="badge-hold">HOLD</span>'
        else:
            return '<span class="badge-sell">SELL</span>'

    # Top KPI Strip
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-label">LATEST CLOSE</div>
            <div class="metric-val">₹{curr_price:,.2f}</div>
            <div class="metric-sub" style="color: {'#10B981' if price_chg >= 0 else '#EF4444'};">
                {'+' if price_chg >= 0 else ''}{price_chg:,.2f} ({price_chg_pct:+.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-label">RL POLICY SIGNAL</div>
            <div style="margin-top: 6px;">{get_badge_html(final_signal)}</div>
            <div class="metric-sub" style="color: #94A3B8;">Action ID: {raw_pred['action_id']}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-label">POLICY CONVICTION</div>
            <div class="metric-val" style="color: #38BDF8;">{conviction*100:.1f}%</div>
            <div class="metric-sub" style="color: #94A3B8;">Categorical Softmax</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-label">HALF-KELLY ALLOC</div>
            <div class="metric-val" style="color: #A855F7;">{alloc_fraction*100:.1f}%</div>
            <div class="metric-sub" style="color: #94A3B8;">Single Stock Cap: 20%</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col5:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-label">CRITIC VALUE V(s)</div>
            <div class="metric-val" style="color: {'#10B981' if raw_pred['critic_value'] >= 0 else '#EF4444'};">
                {raw_pred['critic_value']:+.3f}
            </div>
            <div class="metric-sub" style="color: #94A3B8;">Baseline Advantage</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col6:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-label">95% 1-DAY VAR</div>
            <div class="metric-val" style="color: #F59E0B;">{var_res['var_95_pct']}%</div>
            <div class="metric-sub" style="color: #EF4444;">₹{var_res['var_95_inr']:,.0f} Max 1d Loss</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Main Chart & Multi-Agent Deliberation Layout
    chart_col, right_col = st.columns([1.8, 1.0])

    with chart_col:
        st.markdown("#### Historical Price & Quantitative Risk Channels")
        
        # Interactive Candlestick Chart with Plotly
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=feat_df.index,
                open=feat_df["Open"],
                high=feat_df["High"],
                low=feat_df["Low"],
                close=feat_df["Close"],
                name=f"{selected_symbol} Price",
                increasing_line_color="#10B981",
                decreasing_line_color="#EF4444",
            ),
            row=1, col=1
        )

        # 50-day SMA
        if "SMA_50" in feat_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=feat_df.index,
                    y=feat_df["SMA_50"],
                    name="50-Day SMA",
                    line=dict(color="#38BDF8", width=1.5),
                ),
                row=1, col=1
            )

        # Volatility Bands (Close +/- 2*ATR)
        upper_band = feat_df["Close"] + (2.0 * feat_df["ATRr_14"])
        lower_band = feat_df["Close"] - (2.0 * feat_df["ATRr_14"])

        fig.add_trace(
            go.Scatter(
                x=feat_df.index,
                y=upper_band,
                name="ATR Upper (+2σ)",
                line=dict(color="rgba(245, 158, 11, 0.4)", width=1, dash="dot"),
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=feat_df.index,
                y=lower_band,
                name="ATR Lower (-2σ)",
                line=dict(color="rgba(239, 68, 68, 0.4)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(59, 130, 246, 0.05)",
            ),
            row=1, col=1
        )

        # Volume Bar
        fig.add_trace(
            go.Bar(
                x=feat_df.index,
                y=feat_df["Volume"],
                name="Volume",
                marker_color="rgba(148, 163, 184, 0.3)",
            ),
            row=2, col=1
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.5)",
            margin=dict(l=10, r=10, t=20, b=10),
            height=480,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Backtest Performance Metrics row
        st.markdown("##### 1-Year Deterministic Backtest Performance")
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        b_c1.metric("Total Return", f"{eval_res['total_return']*100:+.2f}%")
        b_c2.metric("Sharpe Ratio", f"{eval_res['sharpe_ratio']:.2f}")
        b_c3.metric("Max Drawdown", f"{eval_res['max_drawdown']*100:.2f}%")
        b_c4.metric("CAGR", f"{eval_res['cagr']*100:+.2f}%")

    with right_col:
        st.markdown("#### Multi-Agent Deliberation Panel")
        
        # Action Probability Distribution
        prob_dist = raw_pred["probability_distribution"]
        prob_df = pd.DataFrame({
            "Action": ["SELL (0%)", "HOLD (25%)", "BUY (65%)", "STRONG BUY (100%)"],
            "Probability": [prob_dist["SELL"], prob_dist["HOLD"], prob_dist["BUY"], prob_dist["STRONG_BUY"]],
            "Color": ["#EF4444", "#F59E0B", "#38BDF8", "#10B981"]
        })

        prob_fig = go.Figure(go.Bar(
            x=prob_df["Probability"] * 100,
            y=prob_df["Action"],
            orientation="h",
            marker=dict(color=prob_df["Color"]),
            text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
            textposition="auto",
        ))
        prob_fig.update_layout(
            title=dict(text="Actor Head Softmax Distribution", font=dict(size=12, color="#94A3B8")),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.3)",
            margin=dict(l=10, r=10, t=30, b=10),
            height=160,
            xaxis=dict(range=[0, 100], showgrid=False),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # Dynamic Volatility Bounds (INR)
        stops = deliberation["agent_deliberations"]["risk"]["stops"]
        stop_loss_inr = stops["stop_loss_inr"]
        take_profit_inr = stops["take_profit_inr"]
        sl_pct = ((curr_price - stop_loss_inr) / curr_price) * 100.0
        tp_pct = ((take_profit_inr - curr_price) / curr_price) * 100.0

        st.markdown(f"""
        <div class="quant-card" style="padding: 16px; margin-bottom: 12px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; margin-bottom: 10px;">
                SEBI DYNAMIC ATR BOUNDS (INR)
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">Trailing Stop-Loss:</span>
                <span style="color: #EF4444; font-weight: 700; font-family: 'JetBrains Mono';">₹{stop_loss_inr:,.2f} (-{sl_pct:.2f}%)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">Take-Profit Target:</span>
                <span style="color: #10B981; font-weight: 700; font-family: 'JetBrains Mono';">₹{take_profit_inr:,.2f} (+{tp_pct:.2f}%)</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span style="color: #94A3B8; font-size: 0.85rem;">Risk / Reward Ratio:</span>
                <span style="color: #F8FAFC; font-weight: 700; font-family: 'JetBrains Mono';">1 : {stops['risk_reward_ratio']:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94A3B8; font-size: 0.85rem;">14-Period ATR:</span>
                <span style="color: #F59E0B; font-weight: 700; font-family: 'JetBrains Mono';">₹{stops['atr_14_inr']:,.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Consensus Audit Note
        st.markdown(f"""
        <div class="quant-card-glass" style="font-size: 0.8rem; color: #CBD5E1; line-height: 1.45;">
            <span style="color: #38BDF8; font-weight: 700;">AUDIT NARRATIVE:</span><br>
            {deliberation['executive_summary']}
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 2: Institutional 10-Stock Portfolio (₹1 Crore Book)
# =============================================================================
with tab2:
    st.markdown("### Multi-Asset Institutional Cross-Sectional Allocation")
    st.markdown("""
    Optimizes a ₹1,00,00,000 (1 Crore INR) institutional equity book across top NSE Large-Caps 
    subject to SEBI exposure limits (20% single-stock ceiling) and dynamic cash preservation.
    """)

    run_sim_btn = st.button("Run / Rebalance Full Portfolio Simulation", type="primary")
    if run_sim_btn:
        st.cache_data.clear()

    @st.cache_data(ttl=1800)
    def compute_portfolio_simulation(epochs=10):
        engine = InstitutionalPortfolioEngine(universe=DEFAULT_UNIVERSE, initial_capital_inr=10_000_000.0)
        return engine.run_multi_asset_portfolio(epochs_per_asset=epochs, cycles_per_asset=2)

    with st.spinner("Computing cross-sectional RL rollouts across 10 NSE assets..."):
        p_res = compute_portfolio_simulation(epochs=10)

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_met = p_res.get("portfolio_metrics", {})

    p_col1.metric("Portfolio Sharpe", f"{p_met.get('portfolio_sharpe_ratio', 0.0):.2f}")
    p_col2.metric("1Y Cumulative Return", f"{p_met.get('portfolio_total_return_pct', 0.0):+.2f}%")
    p_col3.metric("Max Portfolio Drawdown", f"{p_met.get('portfolio_max_drawdown_pct', 0.0):.2f}%")
    p_col4.metric("Calmar Ratio", f"{p_met.get('calmar_ratio', 0.0):.2f}")
    p_col5.metric("Cash Reserve", f"{p_res.get('cash_allocation_pct', 0.0):.1f}%", f"₹{p_res.get('cash_inr', 0.0):,.0f}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Portfolio Charts
    p_chart1, p_chart2 = st.columns([1.2, 1.0])

    with p_chart1:
        # Cumulative Growth Chart
        traj = p_res.get("cumulative_portfolio_trajectory") or p_res.get("portfolio_nav_curve", [1.0])
        if not traj:
            traj = [1.0]
        dates = pd.date_range(end=pd.Timestamp.today(), periods=len(traj), freq="B")

        
        cum_fig = go.Figure()
        cum_fig.add_trace(go.Scatter(
            x=dates,
            y=[v * 100.0 - 100.0 for v in traj],
            name="QuantEdge RL Portfolio",
            line=dict(color="#10B981", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(16, 185, 129, 0.08)",
        ))
        cum_fig.update_layout(
            title="Portfolio Cumulative Growth (%) [Holding Horizon]",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.3)",
            height=320,
            margin=dict(l=10, r=10, t=35, b=10),
            yaxis=dict(title="Return (%)"),
        )
        st.plotly_chart(cum_fig, use_container_width=True)

    with p_chart2:
        # Capital Allocation Donut
        allocs = p_res["current_allocations"]
        donut_labels = list(allocs.keys()) + ["CASH RESERVE (INR)"]
        donut_vals = [allocs[k]["target_weight"] * 100.0 for k in allocs] + [p_res["cash_allocation_pct"]]

        donut_fig = go.Figure(data=[go.Pie(
            labels=donut_labels,
            values=donut_vals,
            hole=0.55,
            marker=dict(colors=px.colors.qualitative.Prism),
        )])
        donut_fig.update_layout(
            title="Capital Distribution (₹1 Crore Book)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.3)",
            height=320,
            margin=dict(l=10, r=10, t=35, b=10),
            showlegend=False,
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    # Detailed Allocation Table
    st.markdown("#### Institutional Execution & Order Routing Matrix")
    rows = []
    for sym, item in allocs.items():
        rows.append({
            "Symbol": sym,
            "Signal": item["signal"],
            "Target Weight": f"{item['target_weight']*100:.2f}%",
            "Allocated Capital (₹)": f"₹{item['capital_inr']:,.2f}",
            "Conviction": f"{item['confidence_pct']:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# =============================================================================
# TAB 3: Deep Explainability Lab
# =============================================================================
with tab3:
    st.markdown("### Gradient Sensitivity & Factor Attribution Lab")
    st.markdown("""
    QuantEdge implements first-order gradient saliency $S_i = \left| \\frac{\\partial \\text{logit}}{\\partial s_i} \\cdot s_i \\right|$ 
    to provide mathematically rigorous explanations for every trading signal, meeting institutional compliance requirements.
    """)

    # Compute attribution for selected stock
    mat = prepare_market_state_matrix(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
    latest_state = np.append(mat[-1], [0.0, 0.0]).astype(np.float32)

    explainer = PolicyExplainer(rl_sys.network)
    exp_res = explainer.explain_signal(latest_state, chosen_action_idx=raw_pred["action_id"])

    exp_col1, exp_col2 = st.columns([1.2, 1.0])

    with exp_col1:
        st.markdown("#### Input Feature Attribution Ranking")
        feat_attr_df = pd.DataFrame(exp_res["feature_attributions"])
        
        attr_fig = go.Figure(go.Bar(
            x=feat_attr_df["attribution_pct"],
            y=feat_attr_df["feature"],
            orientation="h",
            marker=dict(
                color=feat_attr_df["attribution_pct"],
                colorscale="Viridis",
            ),
            text=[f"{val:.2f}%" for val in feat_attr_df["attribution_pct"]],
            textposition="auto",
        ))
        attr_fig.update_layout(
            title="First-Order Gradient Sensitivity (% Contribution)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.3)",
            height=420,
            margin=dict(l=10, r=10, t=35, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(attr_fig, use_container_width=True)

    with exp_col2:
        st.markdown("#### Institutional Factor Decomposition")
        factors = exp_res["factor_decomposition"]
        f_names = list(factors.keys())
        f_weights = list(factors.values())

        f_fig = go.Figure(data=[go.Pie(
            labels=f_names,
            values=f_weights,
            hole=0.5,
            marker=dict(colors=["#38BDF8", "#818CF8", "#F59E0B", "#10B981", "#EC4899"]),
        )])
        f_fig.update_layout(
            title="Macro & Micro Factor Clustering (%)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.3)",
            height=280,
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(f_fig, use_container_width=True)

        st.markdown(f"""
        <div class="quant-card" style="padding: 16px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">
                SEBI GOVERNANCE COMPLIANCE NARRATIVE
            </div>
            <div style="font-size: 0.82rem; color: #E2E8F0; line-height: 1.5;">
                {exp_res['institutional_narrative']}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: Actor-Critic Training Sandbox
# =============================================================================
with tab4:
    st.markdown("### Actor-Critic Hyperparameter Sandbox & Rollout Simulator")
    st.markdown("""
    Explore and fine-tune the core parameters of the foundational Actor-Critic engine.
    Observe real-time policy and critic loss dynamics across rollouts.
    """)

    sb_c1, sb_c2 = st.columns(2)
    with sb_c1:
        custom_lr = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 1.8e-3, 3e-3, 5e-3], value=1.8e-3)
        custom_gamma = st.slider("Discount Factor (γ)", 0.90, 0.999, 0.99, step=0.01)
    with sb_c2:
        custom_downside = st.slider("Downside Semivariance Penalty (λ)", 0.0, 2.0, 0.5, step=0.1)
        custom_friction = st.slider("Statutory Friction (%)", 0.0, 0.5, 0.10, step=0.05) / 100.0

    # Display Architecture Summary
    st.markdown("#### Neural Network Weight Topology (Exact 9,573 Parameters)")
    arch_df = pd.DataFrame([
        {"Layer": "Backbone Linear 1", "Dimensions": "12 → 64", "Activation / Norm": "LayerNorm + LeakyReLU(0.1)", "Parameters": "832 + 128 = 960"},
        {"Layer": "Backbone Linear 2", "Dimensions": "64 → 64", "Activation / Norm": "LayerNorm + LeakyReLU(0.1)", "Parameters": "4,160 + 128 = 4,288"},
        {"Layer": "Actor Policy Head", "Dimensions": "64 → 32 → 4", "Activation / Norm": "LeakyReLU(0.1) + Softmax", "Parameters": "2,080 + 132 = 2,212"},
        {"Layer": "Critic Value Head", "Dimensions": "64 → 32 → 1", "Activation / Norm": "LeakyReLU(0.1) + Linear", "Parameters": "2,080 + 33 = 2,113"},
    ])
    st.dataframe(arch_df, use_container_width=True)

    # Training Loss Curve
    st.markdown("#### Rollout Loss Evolution")
    if train_stats:
        loss_df = pd.DataFrame(train_stats)
        l_fig = go.Figure()
        l_fig.add_trace(go.Scatter(x=loss_df["cycle"], y=loss_df["policy_loss"], name="Actor Policy Loss", line=dict(color="#38BDF8", width=2)))
        l_fig.add_trace(go.Scatter(x=loss_df["cycle"], y=loss_df["value_loss"], name="Critic Value MSE Loss", line=dict(color="#F59E0B", width=2)))
        l_fig.update_layout(
            title="Actor-Critic Optimization Trajectory across Rollout Cycles",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.3)",
            height=280,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(title="Rollout Iteration"),
            yaxis=dict(title="Loss Magnitude"),
        )
        st.plotly_chart(l_fig, use_container_width=True)


# =============================================================================
# TAB 5: Macro Regimes & Sentiment Radar
# =============================================================================
with tab5:
    st.markdown("### Indian Macroeconomic Regimes & Sentiment Radar")
    st.markdown("""
    Indian equity markets are heavily influenced by Foreign Institutional Investor (FII) flows, 
    Domestic Institutional Investor (DII) buying, and volatility shocks on the **INDIA VIX**.
    """)

    sent_res = MarketSentimentEngine.analyze_sentiment(
        symbol=selected_symbol,
        price_change_5d=float(feat_df["Ret_5d"].iloc[-1]),
        vix_level=vix_val,
        nifty_ret=nifty_ret,
    )

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown(f"""
        <div class="quant-card">
            <h4 style="margin: 0 0 12px 0; color: #38BDF8;">INDIA VIX & REGIME SENSITIVITY</h4>
            <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono'; margin-bottom: 8px;">
                {vix_val:.2f} <span style="font-size: 1rem; color: #94A3B8;">points</span>
            </div>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.5;">
                <b>Classified State:</b> <span style="color: {reg_color}; font-weight: 700;">{regime_info['regime']}</span><br>
                <b>Systemic Risk Level:</b> {regime_info['risk_level']}<br>
                <b>Crisis Circuit Breaker:</b> {'ACTIVE [CRISIS MODE]' if regime_info['is_crisis'] else 'INACTIVE (Normal Trading)'}<br>
                <b>Max Permitted Equity Cap:</b> {regime_info['equity_cap']*100:.0f}%
            </p>
            <div style="background: rgba(0,0,0,0.25); padding: 12px; border-radius: 6px; font-size: 0.8rem; color: #94A3B8;">
                {regime_info['description']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="quant-card">
            <h4 style="margin: 0 0 12px 0; color: #10B981;">INSTITUTIONAL FLOW PROXY (FII / DII)</h4>
            <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono'; margin-bottom: 8px;">
                {sent_res['sentiment_label']}
            </div>
            <p style="color: #CBD5E1; font-size: 0.85rem; line-height: 1.5;">
                <b>Sentiment Score:</b> {sent_res['sentiment_score']:+.3f} (Scale: -1.0 to +1.0)<br>
                <b>Flow Posture:</b> {sent_res['institutional_flow_proxy']}<br>
                <b>NIFTY 50 1-Day Benchmark Beta:</b> {nifty_ret*100:+.2f}%<br>
                <b>5-Day Asset Drift:</b> {float(feat_df['Ret_5d'].iloc[-1])*100:+.2f}%
            </p>
            <div style="background: rgba(0,0,0,0.25); padding: 12px; border-radius: 6px; font-size: 0.8rem; color: #94A3B8;">
                Sentiment engine correlates short-term drift with INDIA VIX compression to estimate whether 
                FIIs are net buyers or hedging into index options.
            </div>
        </div>
        """, unsafe_allow_html=True)
