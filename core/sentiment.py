"""
QuantEdge AI — Indian Equity Market Sentiment Module
Synthesizes FII/DII institutional flow indicators, volatility regime sentiment,
and Indian market momentum into an institutional sentiment score.
"""

import numpy as np
from typing import Dict, Any


class MarketSentimentEngine:
    """
    Computes institutional sentiment for Indian equities using:
    - Institutional Flow Proxy (FII / DII net posture)
    - INDIA VIX Fear / Greed gauge
    - Relative performance against benchmark NIFTY 50
    """

    @staticmethod
    def analyze_sentiment(
        symbol: str,
        price_change_5d: float,
        vix_level: float,
        nifty_ret: float,
    ) -> Dict[str, Any]:
        """
        Returns sentiment score in [-1.0, +1.0] and institutional sentiment classification.
        """
        # 1. VIX sentiment: low VIX indicates institutional complacency/greed, high VIX indicates fear
        vix_score = np.clip((18.0 - vix_level) / 10.0, -1.0, 1.0)

        # 2. Benchmark momentum score
        nifty_score = np.clip(nifty_ret * 30.0, -1.0, 1.0)

        # 3. Stock specific momentum score
        stock_score = np.clip(price_change_5d * 15.0, -1.0, 1.0)

        # Composite weighted score
        composite = float(0.4 * stock_score + 0.35 * nifty_score + 0.25 * vix_score)
        composite = float(np.clip(composite, -1.0, 1.0))

        if composite >= 0.35:
            sentiment_label = "BULLISH"
            fii_posture = "NET_BUYERS"
        elif composite <= -0.35:
            sentiment_label = "BEARISH"
            fii_posture = "NET_SELLERS"
        else:
            sentiment_label = "NEUTRAL"
            fii_posture = "BALANCED"

        return {
            "symbol": symbol,
            "sentiment_score": round(composite, 3),
            "sentiment_label": sentiment_label,
            "institutional_flow_proxy": fii_posture,
            "vix_contribution": round(float(vix_score), 3),
            "nifty_contribution": round(float(nifty_score), 3),
        }
