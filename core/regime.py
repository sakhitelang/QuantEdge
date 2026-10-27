"""
QuantEdge AI — Macro Regime Classification Module
Classifies Indian macroeconomic and equity market regimes based on
INDIA VIX levels and NIFTY 50 trend momentum.
"""

from typing import Dict, Any


class MacroRegimeClassifier:
    """
    Classifies Indian equity market into 6 distinct institutional regimes:
    1. CRISIS_PANIC: VIX > 24.0 (Extreme fear, margin compression)
    2. BEAR_TRENDING: VIX > 17.5 and r_nifty < -0.7%
    3. RECOVERY: 16.0 < VIX < 23.0 and r_nifty > +0.8%
    4. BULL_TRENDING: VIX < 14.0 and r_nifty >= 0.0
    5. BULL_VOLATILE: 14.0 <= VIX <= 18.0 and r_nifty > 0.0
    6. SIDEWAYS_RANGE: Neutral range-bound market
    """

    @staticmethod
    def classify_regime(vix: float, nifty_ret: float) -> Dict[str, Any]:
        vix_val = float(vix)
        ret_val = float(nifty_ret)

        if vix_val > 24.0:
            regime = "CRISIS_PANIC"
            risk_level = "CRITICAL"
            equity_cap = 0.15
            description = "Severe macro panic / institutional hedging spike (INDIA VIX > 24.0)."
        elif vix_val > 17.5 and ret_val < -0.007:
            regime = "BEAR_TRENDING"
            risk_level = "HIGH"
            equity_cap = 0.35
            description = "Bearish institutional rotation and broad market sell-off (NIFTY down > 0.7%)."
        elif 16.0 < vix_val < 23.0 and ret_val > 0.008:
            regime = "RECOVERY"
            risk_level = "MODERATE"
            equity_cap = 0.75
            description = "Post-shock short covering and liquidity recovery in progress."
        elif vix_val < 14.0 and ret_val >= 0.0:
            regime = "BULL_TRENDING"
            risk_level = "LOW"
            equity_cap = 1.00
            description = "Low volatility secular bullish expansion (VIX < 14.0, positive momentum)."
        elif 14.0 <= vix_val <= 18.0 and ret_val > 0.0:
            regime = "BULL_VOLATILE"
            risk_level = "MODERATE"
            equity_cap = 0.70
            description = "Constructive bullish trend accompanied by elevated intraday volatility."
        else:
            regime = "SIDEWAYS_RANGE"
            risk_level = "NEUTRAL"
            equity_cap = 0.50
            description = "Range-bound consolidation without clear directional trend."

        return {
            "regime": regime,
            "vix": round(vix_val, 2),
            "nifty_return_pct": round(ret_val * 100.0, 2),
            "risk_level": risk_level,
            "equity_cap": equity_cap,
            "description": description,
            "is_crisis": regime == "CRISIS_PANIC",
        }
