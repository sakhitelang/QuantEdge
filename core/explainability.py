"""
QuantEdge AI — Explainability & Attribution Module
Provides feature attribution, sensitivity analysis, and factor decomposition
to explain Reinforcement Learning policy selections for institutional compliance.
"""

import torch
import numpy as np
from typing import Dict, List, Any
from core.models import ActorCriticNetwork

FEATURE_NAMES = [
    "Ret_1d (1-Day Return)",
    "Ret_5d (5-Day Momentum)",
    "Ret_20d (20-Day Momentum)",
    "Normalized RSI-14",
    "Normalized MACD Hist",
    "Normalized ATR Band",
    "Normalized Realized Vol",
    "Dist from 50-SMA",
    "Normalized INDIA VIX",
    "NIFTY 50 Daily Return",
    "Current Holding Weight",
    "Unrealized Step PnL",
]


class PolicyExplainer:
    """
    Computes input feature attribution and factor decomposition for the PPO Actor network
    using first-order gradient sensitivity: S_i = |d(logit_action)/d(feature_i)|.
    """

    def __init__(self, network: ActorCriticNetwork):
        self.network = network

    def explain_signal(
        self,
        state_vector: np.ndarray,
        chosen_action_idx: int,
        feature_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculates attribution scores for all 12 input features and groups them into
        institutional factor groups (Momentum, Trend, Volatility, Macro, Portfolio).
        """
        names = feature_names or FEATURE_NAMES
        st_t = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0).requires_grad_(True)

        self.network.eval()
        logits, _ = self.network(st_t)
        target_logit = logits[0, chosen_action_idx]

        self.network.zero_grad()
        target_logit.backward()

        grads = st_t.grad.squeeze().detach().cpu().numpy()
        attributions = np.abs(grads * state_vector)
        total_attr = np.sum(attributions) + 1e-8
        norm_attributions = attributions / total_attr

        # Feature level ranking
        ranked_features = []
        for i in range(len(names)):
            ranked_features.append({
                "feature": names[i],
                "value": round(float(state_vector[i]), 4),
                "gradient": round(float(grads[i]), 4),
                "attribution_pct": round(float(norm_attributions[i] * 100.0), 2),
            })

        ranked_features.sort(key=lambda x: x["attribution_pct"], reverse=True)

        # Factor decomposition
        # Indices:
        # Momentum: 0, 1, 2
        # Trend: 4, 7
        # Volatility: 3, 5, 6
        # Macro: 8, 9
        # Portfolio: 10, 11
        factor_scores = {
            "Momentum Factors": float(np.sum(norm_attributions[[0, 1, 2]]) * 100.0),
            "Trend & Moving Averages": float(np.sum(norm_attributions[[4, 7]]) * 100.0),
            "Volatility & Risk Bands": float(np.sum(norm_attributions[[3, 5, 6]]) * 100.0),
            "Macro Regimes (VIX & NIFTY)": float(np.sum(norm_attributions[[8, 9]]) * 100.0),
            "Internal Portfolio State": float(np.sum(norm_attributions[[10, 11]]) * 100.0),
        }

        # Formulate executive institutional narrative
        top_factor = max(factor_scores.items(), key=lambda x: x[1])
        top_feature = ranked_features[0]["feature"]
        narrative = (
            f"Policy selection is primarily governed by '{top_factor[0]}' ({top_factor[1]:.1f}% weight), "
            f"with the strongest individual sensitivity driven by {top_feature} "
            f"({ranked_features[0]['attribution_pct']}%), ensuring mathematically grounded risk capture."
        )

        return {
            "top_factor": top_factor[0],
            "top_factor_weight_pct": round(top_factor[1], 2),
            "factor_decomposition": {k: round(v, 2) for k, v in factor_scores.items()},
            "feature_attributions": ranked_features,
            "institutional_narrative": narrative,
        }
