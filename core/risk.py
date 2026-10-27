"""
QuantEdge AI — Risk Governance & Position Sizing Module
Implements Half-Kelly Capital Allocation, ATR-based Volatility Stops in INR,
95% Daily Value-at-Risk (VaR), and Crisis Regime Overrides.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class RiskGovernor:
    """
    Institutional Risk Management Layer for Indian Equities (SEBI compliant):
    - Half-Kelly Position Sizing with 20% single-stock hard limit
    - Dynamic ATR Volatility Trailing Stops & Targets in INR
    - 95% 1-day Historical & Parametric Value at Risk (VaR)
    - Crisis Regime safety interlock (emergency deleveraging)
    """

    MAX_SINGLE_STOCK_ALLOCATION = 0.20  # 20% max single-stock exposure

    @classmethod
    def calculate_half_kelly(
        cls,
        expected_return: float,
        downside_risk: float,
        confidence: float,
    ) -> float:
        """
        Half-Kelly Criterion:
        f* = (b * p - q) / (2 * b)
        where:
          b = payoff ratio (|expected_return / downside_risk|)
          p = win probability parameterized by confidence (0.5 + confidence / 2.0)
          q = 1 - p
        """
        b = abs(expected_return / (downside_risk + 1e-6))
        b = max(0.5, min(4.0, b))  # Cap payoff ratio to sensible bounds

        # Scale confidence (typically 0.4 to 0.95) to probability space [0.5, 0.975]
        p = 0.5 + (float(confidence) / 2.0)
        p = max(0.40, min(0.95, p))
        q = 1.0 - p

        raw_kelly = (b * p - q) / (b + 1e-8)
        half_kelly = raw_kelly / 2.0
        return float(max(0.0, min(cls.MAX_SINGLE_STOCK_ALLOCATION, half_kelly)))

    @classmethod
    def compute_position_size(
        cls,
        rl_target_weight: float,
        confidence: float,
        expected_return: float = 0.03,
        downside_risk: float = 0.015,
        regime_cap: float = 1.0,
    ) -> float:
        """
        Blends Half-Kelly sizing with the RL target weight,
        bounded by institutional single-stock cap (20%) and macro regime cap.
        """
        if rl_target_weight <= 0.0:
            return 0.0

        half_kelly = cls.calculate_half_kelly(expected_return, downside_risk, confidence)
        rl_fraction = rl_target_weight * cls.MAX_SINGLE_STOCK_ALLOCATION

        # 50/50 blend of Half-Kelly and RL conviction
        blended = 0.5 * half_kelly + 0.5 * rl_fraction
        capped = min(cls.MAX_SINGLE_STOCK_ALLOCATION, blended)
        final_size = capped * min(1.0, regime_cap)

        return float(round(final_size, 4))

    @staticmethod
    def compute_atr_stops(
        current_price: float,
        atr_14: float,
        stop_mult: float = 2.0,
        target_mult: float = 3.5,
    ) -> Dict[str, float]:
        """
        Calculates ATR-based volatility stops and profit targets in Indian Rupees (INR).
        - Stop Loss = Price - 2.0 * ATR14
        - Take Profit = Price + 3.5 * ATR14
        """
        p = float(current_price)
        a = float(atr_14) if atr_14 > 0 else (p * 0.015)

        stop_loss = max(1.0, p - (stop_mult * a))
        take_profit = p + (target_mult * a)
        risk_per_share = p - stop_loss
        reward_per_share = take_profit - p
        risk_reward_ratio = (reward_per_share / (risk_per_share + 1e-8))

        return {
            "current_price_inr": round(p, 2),
            "atr_14_inr": round(a, 2),
            "stop_loss_inr": round(stop_loss, 2),
            "take_profit_inr": round(take_profit, 2),
            "risk_per_share_inr": round(risk_per_share, 2),
            "risk_reward_ratio": round(risk_reward_ratio, 2),
        }

    @staticmethod
    def compute_daily_var_95(returns_series: np.ndarray, portfolio_capital_inr: float = 1_000_000.0) -> Dict[str, float]:
        """
        Calculates 1-day 95% Value at Risk (VaR) both Parametrically and Historically.
        """
        rets = np.array(returns_series, dtype=np.float64)
        rets = rets[~np.isnan(rets)]
        if len(rets) < 10:
            return {"var_95_pct": 2.5, "var_95_inr": portfolio_capital_inr * 0.025}

        # 1. Historical VaR at 5th percentile
        hist_var_pct = abs(float(np.percentile(rets, 5)))

        # 2. Parametric VaR (1.65 * standard deviation)
        mu = float(np.mean(rets))
        sigma = float(np.std(rets))
        param_var_pct = abs(mu - 1.65 * sigma)

        # Conservative selection
        var_pct = max(hist_var_pct, param_var_pct)
        var_inr = portfolio_capital_inr * var_pct

        return {
            "var_95_pct": round(var_pct * 100.0, 2),
            "var_95_inr": round(var_inr, 2),
            "historical_var_pct": round(hist_var_pct * 100.0, 2),
            "parametric_var_pct": round(param_var_pct * 100.0, 2),
        }

    @classmethod
    def apply_crisis_override(
        cls,
        rl_signal: str,
        confidence: float,
        suggested_size: float,
        is_crisis: bool,
    ) -> Tuple[str, float, float, Optional[str]]:
        """
        Crisis Regime Safety Interlock:
        If market is in CRISIS_PANIC (VIX > 24), overrides BUY/STRONG_BUY to HOLD,
        caps sizing to 15%, and discounts model confidence by 30%.
        """
        if not is_crisis:
            return rl_signal, confidence, suggested_size, None

        override_msg = "CRISIS INTERLOCK ACTIVATED: INDIA VIX > 24.0. Action forced to HOLD, allocation capped at 15%."
        discounted_conf = float(confidence * 0.70)
        forced_signal = "HOLD" if rl_signal in ["BUY", "STRONG_BUY"] else rl_signal
        capped_size = min(0.15, suggested_size)

        return forced_signal, discounted_conf, capped_size, override_msg
