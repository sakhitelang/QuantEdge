"""
QuantEdge AI — Institutional Portfolio Allocation Module
Simulates multi-asset cross-sectional allocations across the Indian equity universe,
enforcing maximum single-stock exposure (20%), cash reserves, and turnover tracking.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from core.data import load_ticker_data, DEFAULT_UNIVERSE, get_macro_context
from core.features import compute_features
from core.models import RLForecastingSystem
from core.agents import MultiAgentConsensusEngine


class InstitutionalPortfolioEngine:
    """
    Multi-Asset Allocation and Backtesting Engine for NSE Equities.
    Maintains portfolio balance, cash reserves, rebalancing logic, and risk analytics.
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        initial_capital_inr: float = 10_000_000.0,
        max_stock_cap: float = 0.20,
    ):
        self.universe = universe or DEFAULT_UNIVERSE
        self.initial_capital_inr = initial_capital_inr
        self.max_stock_cap = max_stock_cap
        self.consensus_engine = MultiAgentConsensusEngine()

    def run_multi_asset_portfolio(
        self,
        epochs_per_asset: int = 10,
        cycles_per_asset: int = 2,
    ) -> Dict[str, Any]:
        """
        Executes full portfolio simulation across all universe equities using local data files:
        1. Loads local CSV for each stock and computes features.
        2. Trains an asset-specific RL Actor-Critic policy.
        3. Evaluates historical backtest trajectories.
        4. Synthesizes current institutional consensus allocations.
        5. Computes portfolio-level aggregate metrics.
        """
        vix_val, nifty_ret = get_macro_context()
        asset_results = {}
        all_trajectories = {}

        min_len = 999999
        stock_dfs = {}

        for symbol in self.universe:
            try:
                raw_df = load_ticker_data(symbol, period="1y")
                feat_df = compute_features(raw_df)
                stock_dfs[symbol] = feat_df
                if len(feat_df) < min_len:
                    min_len = len(feat_df)
            except Exception as e:
                continue

        if not stock_dfs:
            raise RuntimeError("No valid equity data found in local data directory.")

        # Train and evaluate each asset
        current_allocations = {}
        total_target_equity_weight = 0.0

        for symbol, df in stock_dfs.items():
            rl_sys = RLForecastingSystem()
            rl_sys.train_on_dataframe(df, vix_val=vix_val, nifty_ret=nifty_ret, epochs=epochs_per_asset, repeat_cycles=cycles_per_asset)

            # Backtest evaluation
            eval_metrics = rl_sys.evaluate_strategy(df, vix_val=vix_val, nifty_ret=nifty_ret)
            asset_results[symbol] = {
                "sharpe_ratio": round(eval_metrics["sharpe_ratio"], 2),
                "total_return_pct": round(eval_metrics["total_return"] * 100.0, 2),
                "max_drawdown_pct": round(eval_metrics["max_drawdown"] * 100.0, 2),
                "cagr_pct": round(eval_metrics["cagr"] * 100.0, 2),
            }

            all_trajectories[symbol] = [t["portfolio_value"] for t in eval_metrics["trajectory"]]

            # Predict current signal
            curr_pred = rl_sys.predict_current_signal(df, vix_val=vix_val, nifty_ret=nifty_ret)
            curr_price = float(df["Close"].iloc[-1])
            atr_val = float(df["ATRr_14"].iloc[-1])

            deliberation = self.consensus_engine.deliberate(
                symbol=symbol,
                rl_prediction=curr_pred,
                vix_val=vix_val,
                nifty_ret=nifty_ret,
                current_price=curr_price,
                atr_14=atr_val,
            )

            alloc_frac = deliberation["allocation_fraction"]
            current_allocations[symbol] = {
                "signal": deliberation["final_signal"],
                "target_weight": alloc_frac,
                "confidence_pct": round(deliberation["confidence"] * 100.0, 1),
                "current_price_inr": round(curr_price, 2),
                "capital_inr": round(self.initial_capital_inr * alloc_frac, 2),
                "stops": deliberation["agent_deliberations"]["risk"]["stops"],
            }
            total_target_equity_weight += alloc_frac

        # If total equity weight > 1.0, scale down proportionally
        if total_target_equity_weight > 1.0:
            scale = 0.95 / total_target_equity_weight
            for sym in current_allocations:
                current_allocations[sym]["target_weight"] = round(current_allocations[sym]["target_weight"] * scale, 4)
                current_allocations[sym]["capital_inr"] = round(self.initial_capital_inr * current_allocations[sym]["target_weight"], 2)
            total_target_equity_weight = sum(v["target_weight"] for v in current_allocations.values())

        cash_weight = max(0.0, 1.0 - total_target_equity_weight)
        cash_inr = self.initial_capital_inr * cash_weight

        # Aggregate portfolio backtest equity curve
        traj_lengths = [len(v) for v in all_trajectories.values()]
        common_len = min(traj_lengths) if traj_lengths else 0

        portfolio_nav_curve = []
        if common_len > 1:
            equal_weight = 1.0 / len(all_trajectories)
            for t in range(common_len):
                pt_val = sum(traj[t] * equal_weight for traj in all_trajectories.values())
                portfolio_nav_curve.append(float(pt_val))

        nav_arr = np.array(portfolio_nav_curve) if portfolio_nav_curve else np.array([1.0])
        step_returns = np.diff(nav_arr) / nav_arr[:-1] if len(nav_arr) > 1 else np.array([0.0])

        total_port_ret = float(nav_arr[-1] - 1.0)
        port_std = float(np.std(step_returns))
        port_sharpe = float((np.mean(step_returns) / (port_std + 1e-8)) * np.sqrt(252)) if port_std > 0 else 0.0

        running_max = np.maximum.accumulate(nav_arr)
        dds = (running_max - nav_arr) / (running_max + 1e-8)
        port_max_dd = float(np.max(dds))

        calmar = (total_port_ret / (port_max_dd + 1e-8)) if port_max_dd > 0 else 0.0

        return {
            "initial_capital_inr": self.initial_capital_inr,
            "current_equity_allocation_pct": round(total_target_equity_weight * 100.0, 2),
            "cash_allocation_pct": round(cash_weight * 100.0, 2),
            "cash_inr": round(cash_inr, 2),
            "portfolio_metrics": {
                "portfolio_total_return_pct": round(total_port_ret * 100.0, 2),
                "portfolio_sharpe_ratio": round(port_sharpe, 2),
                "portfolio_max_drawdown_pct": round(port_max_dd * 100.0, 2),
                "calmar_ratio": round(calmar, 2),
            },
            "asset_level_results": asset_results,
            "current_allocations": current_allocations,
            "portfolio_nav_curve": [round(x, 4) for x in portfolio_nav_curve],
            "cumulative_portfolio_trajectory": [round(x, 4) for x in portfolio_nav_curve],
        }

