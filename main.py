#!/usr/bin/env python3
"""
QuantEdge AI — Institutional Indian Stock Market (NSE / BSE) Quantitative RL Engine
Executes end-to-end data download to local CSVs, quantitative feature computation,
Actor-Critic reinforcement learning training, multi-agent consensus, and risk governance.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data import (
    load_ticker_data,
    download_all_market_data,
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
)
from core.regime import MacroRegimeClassifier
from core.risk import RiskGovernor
from core.sentiment import MarketSentimentEngine
from core.explainability import PolicyExplainer
from core.agents import MultiAgentConsensusEngine
from core.portfolio import InstitutionalPortfolioEngine


def print_banner():
    banner = """
========================================================================================
   ____                   _   ______     _               ___    ____ 
  / __ \\__  ______ _____  / |_/ / ____/___/ /___ ____      /   |  /  _/ 
 / / / / / / / __ `/ __ \\/ __/ / __/ / __  / __ `/ _ \\    / /| |  / /   
/ /_/ / /_/ / /_/ / / / / /_/ / /___/ /_/ / /_/ /  __/   / ___ |_/ /    
\\___\\_\\__,_/\\__,_/_/ /_/\\__/_/_____/\\__,_/\\__, /\\___/   /_/  |_/___/    
       INDIAN STOCK MARKET (NSE / BSE)   /____/  ACTOR-CRITIC ENGINE 
========================================================================================
    """
    print(banner)


def handle_download(universe, period="1y", interval="1d"):
    print(f"\n[1/3] Downloading Indian Equity Universe from Yahoo Finance directly to local CSVs...")
    print(f"[*] Target Directory: {DATA_DIR}")
    print(f"[*] Universe: {len(universe)} Equities + {len(MACRO_TICKERS)} Macro Indices")
    
    results = download_all_market_data(universe=universe, period=period, interval=interval, force=True)
    for sym, path in results.items():
        fname = os.path.basename(path)
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        print(f"    ✔ Downloaded & Saved: {sym:<15} -> data/{fname} ({sz:,} bytes)")
    print("[*] All datasets downloaded successfully to local storage.\n")


def run_single_stock_pipeline(symbol: str, epochs: int = 15, explain: bool = True):
    clean_sym = normalize_ticker(symbol)
    print(f"\n========================================================================================")
    print(f" [ANALYSIS] Processing Ticker: {clean_sym}")
    print(f"========================================================================================")

    # 1. Ingestion from local CSV (download if not present)
    print(f"\n[Step 1] Loading market OHLCV data from local CSV...")
    df = load_ticker_data(clean_sym, period="1y", interval="1d")
    print(f"    ✔ Local CSV Data Loaded: {len(df)} daily trading sessions")
    print(f"    ✔ Latest Date: {df.index[-1].strftime('%Y-%m-%d')} | Latest Close: ₹{df['Close'].iloc[-1]:,.2f}")

    # 2. Quantitative Feature Computation
    print(f"\n[Step 2] Computing quantitative features with strict temporal causality...")
    feat_df = compute_features(df)
    print(f"    ✔ Features Computed: RSI-14, MACD(12,26,9), ATR-14, 20d Realized Vol, 50d SMA Dist")
    print(f"    ✔ Total Columns: {feat_df.shape[1]} (Shape: {feat_df.shape})")

    # 3. Macro Regime Ingestion
    print(f"\n[Step 3] Loading Indian Macroeconomic Context (^INDIAVIX, ^NSEI)...")
    vix_val, nifty_ret = get_macro_context()
    regime_info = MacroRegimeClassifier.classify_regime(vix_val, nifty_ret)
    print(f"    ✔ INDIA VIX: {regime_info['vix']} | NIFTY 1-Day Return: {regime_info['nifty_return_pct']:+.2f}%")
    print(f"    ✔ Classified Regime: [{regime_info['regime']}] (Risk Level: {regime_info['risk_level']})")
    print(f"    ✔ Regime Policy: {regime_info['description']}")

    # 4. Actor-Critic Neural Training
    print(f"\n[Step 4] Initializing Deep Actor-Critic Network (9,573 Parameters) & Optimizer...")
    rl_sys = RLForecastingSystem()
    param_count = rl_sys.count_parameters()
    print(f"    ✔ Exact Trainable Parameter Count: {param_count:,} weights & biases")
    print(f"    ✔ Executing Actor-Critic rollouts (Epochs: {epochs}, gamma=0.99)...")

    train_stats = rl_sys.train_on_dataframe(feat_df, vix_val=vix_val, nifty_ret=nifty_ret, epochs=epochs, repeat_cycles=2)
    for stat in train_stats:
        print(f"    ✔ Rollout Cycle {stat['cycle']}: Policy Loss = {stat['policy_loss']:.4f} | Critic Loss = {stat['value_loss']:.4f}")

    # 5. Deterministic Historical Backtest
    print(f"\n[Step 5] Backtesting Deterministic Policy across 252 Historical Trading Sessions...")
    eval_res = rl_sys.evaluate_strategy(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
    print(f"    ✔ Backtest Total Return: {eval_res['total_return']*100:+.2f}%")
    print(f"    ✔ Annualized Sharpe Ratio: {eval_res['sharpe_ratio']:.2f}")
    print(f"    ✔ Maximum Peak-to-Trough Drawdown: {eval_res['max_drawdown']*100:.2f}%")
    print(f"    ✔ Compound Annual Growth Rate (CAGR): {eval_res['cagr']*100:+.2f}%")

    # 6. Current State Inference & Consensus Deliberation
    print(f"\n[Step 6] Running Multi-Agent Consensus Deliberation...")
    raw_pred = rl_sys.predict_current_signal(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
    curr_price = float(feat_df["Close"].iloc[-1])
    atr_val = float(feat_df["ATRr_14"].iloc[-1])

    consensus_engine = MultiAgentConsensusEngine()
    deliberation = consensus_engine.deliberate(
        symbol=clean_sym,
        rl_prediction=raw_pred,
        vix_val=vix_val,
        nifty_ret=nifty_ret,
        current_price=curr_price,
        atr_14=atr_val,
    )

    print("\n" + "=" * 80)
    print(f"              INSTITUTIONAL TRADE DECISION REPORT: {clean_sym}")
    print("=" * 80)
    print(f"  RECOMMENDED SIGNAL     : {deliberation['final_signal']}")
    print(f"  TARGET ALLOCATION      : {deliberation['allocation_fraction']*100:.1f}% of Single-Asset Capital")
    print(f"  POLICY CONVICTION      : {deliberation['confidence']*100:.1f}%")
    print(f"  CURRENT ASSET PRICE    : ₹{curr_price:,.2f}")
    print(f"  CRITIC VALUE V(s)      : {raw_pred['critic_value']:+.4f}")
    print(f"  ACTION PROBABILITIES   : SELL={raw_pred['probability_distribution']['SELL']*100:.1f}%, "
          f"HOLD={raw_pred['probability_distribution']['HOLD']*100:.1f}%, "
          f"BUY={raw_pred['probability_distribution']['BUY']*100:.1f}%, "
          f"STRONG_BUY={raw_pred['probability_distribution']['STRONG_BUY']*100:.1f}%")
    print("-" * 80)

    # ATR Stops in INR
    stops = deliberation["agent_deliberations"]["risk"]["stops"]
    print(f"  DYNAMIC VOLATILITY BOUNDS (SEBI RISK GOVERNANCE):")
    print(f"    - Current Price      : ₹{stops['current_price_inr']:,.2f}")
    print(f"    - 14-period ATR      : ₹{stops['atr_14_inr']:,.2f}")
    print(f"    - Trailing Stop-Loss : ₹{stops['stop_loss_inr']:,.2f} (-{((curr_price - stops['stop_loss_inr'])/curr_price)*100:.2f}%)")
    print(f"    - Take-Profit Target : ₹{stops['take_profit_inr']:,.2f} (+{((stops['take_profit_inr'] - curr_price)/curr_price)*100:.2f}%)")
    print(f"    - Risk-Reward Ratio  : 1 : {stops['risk_reward_ratio']:.2f}")
    print("-" * 80)

    # 95% Daily VaR
    var_res = RiskGovernor.compute_daily_var_95(feat_df["Ret_1d"].values, portfolio_capital_inr=1_000_000.0)
    print(f"  PORTFOLIO VALUE AT RISK (95% Daily VaR on ₹10 Lakhs):")
    print(f"    - 1-Day 95% VaR       : {var_res['var_95_pct']}% (₹{var_res['var_95_inr']:,.2f})")
    print(f"    - Historical 5th Pct : {var_res['historical_var_pct']}%")
    print(f"    - Parametric (1.65σ) : {var_res['parametric_var_pct']}%")
    print("-" * 80)

    # Sentiment Breakdown
    sent_res = MarketSentimentEngine.analyze_sentiment(
        symbol=clean_sym,
        price_change_5d=float(feat_df["Ret_5d"].iloc[-1]),
        vix_level=vix_val,
        nifty_ret=nifty_ret,
    )
    print(f"  MARKET SENTIMENT & FLOWS:")
    print(f"    - Sentiment Posture  : {sent_res['sentiment_label']} (Score: {sent_res['sentiment_score']:+.3f})")
    print(f"    - FII / DII Flow Est : {sent_res['institutional_flow_proxy']}")
    print("-" * 80)
    print(f"  EXECUTIVE AUDIT SUMMARY:")
    print(f"    {deliberation['executive_summary']}")
    print("=" * 80)

    # 7. Explainability & Factor Decomposition
    if explain:
        print(f"\n[Step 7] Generating Deep Neural Attribution & Factor Decomposition...")
        mat = prepare_market_state_matrix(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
        latest_state = np.append(mat[-1], [0.0, 0.0]).astype(np.float32)

        explainer = PolicyExplainer(rl_sys.network)
        exp_res = explainer.explain_signal(latest_state, chosen_action_idx=raw_pred["action_id"])

        print(f"\n  FACTOR DECOMPOSITION WEIGHTS:")
        for factor, weight in exp_res["factor_decomposition"].items():
            bar = "█" * int(weight / 4)
            print(f"    - {factor:<30}: {weight:>5.1f}% | {bar}")

        print(f"\n  TOP INFLUENTIAL FEATURES:")
        for feat in exp_res["feature_attributions"][:5]:
            print(f"    - {feat['feature']:<30}: Attribution = {feat['attribution_pct']:>5.2f}% (Val: {feat['value']:+.4f})")
        print(f"\n  GOVERNANCE NARRATIVE: {exp_res['institutional_narrative']}\n")


def run_portfolio_pipeline(epochs: int = 10):
    print(f"\n========================================================================================")
    print(f" [PORTFOLIO SIMULATION] 10-Stock Indian Large-Cap Universe")
    print(f"========================================================================================")
    print(f"[*] Capital Baseline: ₹1,00,00,000 (1 Crore INR Institutional Book)")
    print(f"[*] Regulatory Constraints: 20% Hard Limit per stock, Dynamic Cash Reserve")

    engine = InstitutionalPortfolioEngine(universe=DEFAULT_UNIVERSE, initial_capital_inr=10_000_000.0)
    res = engine.run_multi_asset_portfolio(epochs_per_asset=epochs, cycles_per_asset=2)

    print("\n" + "=" * 80)
    print("              INSTITUTIONAL PORTFOLIO ALLOCATION MATRIX")
    print("=" * 80)
    print(f"{'SYMBOL':<15} | {'SIGNAL':<10} | {'WEIGHT (%)':<12} | {'CAPITAL (INR)':<15} | {'CONFIDENCE':<10}")
    print("-" * 80)

    for sym, alloc in res["current_allocations"].items():
        print(f"{sym:<15} | {alloc['signal']:<10} | {alloc['target_weight']*100:>10.2f}% | ₹{alloc['capital_inr']:>13,.2f} | {alloc['confidence_pct']:>8.1f}%")

    print("-" * 80)
    print(f"  TOTAL EQUITIES ALLOCATION : {res['current_equity_allocation_pct']:.2f}%")
    print(f"  CASH RESERVE (INR)        : {res['cash_allocation_pct']:.2f}% (₹{res['cash_inr']:,.2f})")
    print("=" * 80)

    print(f"\n  BACKTEST PERFORMANCE METRICS (1-Year Holding Horizon):")
    p_met = res["portfolio_metrics"]
    print(f"    - Portfolio Sharpe Ratio    : {p_met['portfolio_sharpe_ratio']:.2f}")
    print(f"    - Portfolio Total Return    : {p_met['portfolio_total_return_pct']:+.2f}%")
    print(f"    - Max Peak-to-Trough DD     : {p_met['portfolio_max_drawdown_pct']:.2f}%")
    print(f"    - Calmar Ratio              : {p_met['calmar_ratio']:.2f}")
    print("=" * 80 + "\n")


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="QuantEdge AI — Institutional Indian Stock Market RL Engine")
    parser.add_argument("--download", action="store_true", help="Download all universe stock CSVs from Yahoo Finance to local data/ folder")
    parser.add_argument("--download-only", action="store_true", help="Download datasets to data/ and exit")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="NSE Ticker to analyze (e.g. RELIANCE.NS, TCS.NS, INFY.NS)")
    parser.add_argument("--portfolio", action="store_true", help="Run full 10-stock institutional portfolio simulation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of Actor-Critic training epochs per cycle")
    parser.add_argument("--explain", action="store_true", default=True, help="Compute deep neural attribution and explainability")

    args = parser.parse_args()

    # Step 1: Handle explicit download request
    if args.download or args.download_only:
        handle_download(universe=DEFAULT_UNIVERSE)
        if args.download_only:
            sys.exit(0)

    # Step 2: Handle Portfolio Simulation or Single Asset Analysis
    if args.portfolio:
        run_portfolio_pipeline(epochs=args.epochs)
    else:
        run_single_stock_pipeline(symbol=args.symbol, epochs=args.epochs, explain=args.explain)


if __name__ == "__main__":
    main()
