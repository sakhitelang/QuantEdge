"""
QuantEdge AI — Multi-Agent Consensus System
Orchestrates autonomous agents (Policy Actor, Value Critic, Macro Regime, Risk Governor)
to form institutional trading consensus for Indian equities.
"""

from typing import Dict, Any, List
from core.regime import MacroRegimeClassifier
from core.risk import RiskGovernor


class RLPolicyActorAgent:
    """Agent representing the Actor policy head."""

    def evaluate(self, rl_prediction: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": "RL_Policy_Actor",
            "action": rl_prediction["action"],
            "target_weight": rl_prediction["target_weight"],
            "confidence": rl_prediction["confidence"],
            "recommendation": f"Allocate {int(rl_prediction['target_weight']*100)}% based on Actor policy head.",
        }


class RLValueCriticAgent:
    """Agent representing the Critic baseline value estimator."""

    def evaluate(self, critic_value: float) -> Dict[str, Any]:
        posture = "BULLISH_ADVANTAGE" if critic_value > 0.05 else ("BEARISH_ADVANTAGE" if critic_value < -0.05 else "NEUTRAL_BASE")
        return {
            "agent": "RL_Value_Critic",
            "critic_state_value": round(critic_value, 4),
            "state_valuation": posture,
            "comment": f"Critic baseline estimate V(s) = {critic_value:+.4f} against rolling risk neutral target.",
        }


class MacroRegimeAgent:
    """Agent tracking Indian macro indices (^INDIAVIX, ^NSEI)."""

    def evaluate(self, vix: float, nifty_ret: float) -> Dict[str, Any]:
        regime_info = MacroRegimeClassifier.classify_regime(vix, nifty_ret)
        return {
            "agent": "Macro_Regime_Governor",
            "regime": regime_info["regime"],
            "vix": regime_info["vix"],
            "nifty_ret_pct": regime_info["nifty_return_pct"],
            "risk_level": regime_info["risk_level"],
            "equity_cap": regime_info["equity_cap"],
            "is_crisis": regime_info["is_crisis"],
            "comment": regime_info["description"],
        }


class RiskGovernorAgent:
    """Agent enforcing SEBI regulatory exposure limits, Half-Kelly sizing, and crisis vetoes."""

    def evaluate(
        self,
        actor_action: str,
        actor_weight: float,
        actor_conf: float,
        is_crisis: bool,
        current_price: float,
        atr_14: float,
        regime_cap: float,
    ) -> Dict[str, Any]:
        # Half-Kelly position size
        suggested_size = RiskGovernor.compute_position_size(
            rl_target_weight=actor_weight,
            confidence=actor_conf,
            regime_cap=regime_cap,
        )

        # ATR stops in INR
        stops = RiskGovernor.compute_atr_stops(current_price, atr_14)

        # Crisis safety check
        final_signal, adj_conf, final_size, override_msg = RiskGovernor.apply_crisis_override(
            rl_signal=actor_action,
            confidence=actor_conf,
            suggested_size=suggested_size,
            is_crisis=is_crisis,
        )

        return {
            "agent": "Risk_Governor",
            "approved_signal": final_signal,
            "approved_allocation": final_size,
            "adjusted_confidence": round(adj_conf, 3),
            "stops": stops,
            "has_crisis_override": bool(override_msg),
            "override_note": override_msg or "Approved within risk limits.",
        }


class MultiAgentConsensusEngine:
    """
    Coordinates deliberation across all 4 agents to produce a unified,
    institutional trade decision with full auditability.
    """

    def __init__(self):
        self.actor_agent = RLPolicyActorAgent()
        self.critic_agent = RLValueCriticAgent()
        self.regime_agent = MacroRegimeAgent()
        self.risk_agent = RiskGovernorAgent()

    def deliberate(
        self,
        symbol: str,
        rl_prediction: Dict[str, Any],
        vix_val: float,
        nifty_ret: float,
        current_price: float,
        atr_14: float,
    ) -> Dict[str, Any]:
        # 1. Run all individual evaluations
        actor_res = self.actor_agent.evaluate(rl_prediction)
        critic_res = self.critic_agent.evaluate(rl_prediction["critic_value"])
        regime_res = self.regime_agent.evaluate(vix_val, nifty_ret)
        risk_res = self.risk_agent.evaluate(
            actor_action=actor_res["action"],
            actor_weight=actor_res["target_weight"],
            actor_conf=actor_res["confidence"],
            is_crisis=regime_res["is_crisis"],
            current_price=current_price,
            atr_14=atr_14,
            regime_cap=regime_res["equity_cap"],
        )

        # 2. Institutional Consensus Formulation
        final_signal = risk_res["approved_signal"]
        final_allocation = risk_res["approved_allocation"]
        final_confidence = risk_res["adjusted_confidence"]

        consensus_agreed = (actor_res["action"] == final_signal)
        action_verb = "EXECUTED" if final_signal in ["BUY", "STRONG_BUY"] else ("LIQUIDATED" if final_signal == "SELL" else "HELD_IN_CASH")

        narrative = (
            f"Consensus reached for {symbol}: Action [{final_signal}] with capital allocation "
            f"of {final_allocation * 100:.1f}% ({final_confidence * 100:.1f}% conviction). "
            f"Regime: {regime_res['regime']} (INDIA VIX: {regime_res['vix']:.1f}). "
            f"{risk_res['override_note']}"
        )

        return {
            "symbol": symbol,
            "final_signal": final_signal,
            "allocation_fraction": final_allocation,
            "confidence": final_confidence,
            "consensus_unanimous": consensus_agreed,
            "executive_summary": narrative,
            "agent_deliberations": {
                "actor": actor_res,
                "critic": critic_res,
                "regime": regime_res,
                "risk": risk_res,
            },
        }
