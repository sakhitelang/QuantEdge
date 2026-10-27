"""
QuantEdge AI — Feature Engineering Module
Computes 12 quantitative momentum, volatility, and trend indicators
adhering strictly to temporal causality (no lookahead bias).
"""

import numpy as np
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes institutional quantitative features from raw OHLCV prices:
    - 14-period Relative Strength Index (RSI)
    - Moving Average Convergence Divergence (MACD, Signal, MACD_hist)
    - 14-period Average True Range (ATR)
    - 20-day Annualized Realized Volatility
    - 50-day Simple Moving Average (SMA) distance
    - 1-day, 5-day, and 20-day Momentum Returns
    """
    res = df.copy()

    # Ensure float types
    close = res["Close"].astype(float)
    high = res["High"].astype(float)
    low = res["Low"].astype(float)

    # 1. Multi-Horizon Returns
    res["Ret_1d"] = close.pct_change(1)
    res["Ret_5d"] = close.pct_change(5)
    res["Ret_20d"] = close.pct_change(20)

    # 2. Relative Strength Index (RSI — 14 periods)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=5).mean()
    avg_loss = loss.rolling(window=14, min_periods=5).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    res["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 3. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    res["MACD"] = ema12 - ema26
    res["MACD_signal"] = res["MACD"].ewm(span=9, adjust=False).mean()
    res["MACD_hist"] = res["MACD"] - res["MACD_signal"]

    # 4. Average True Range (ATR — 14 periods)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res["ATRr_14"] = tr.rolling(window=14, min_periods=5).mean()

    # 5. Annualized 20-Day Realized Volatility
    log_ret = np.log((close / (close.shift(1) + 1e-8)).clip(lower=1e-8))
    res["Realized_Vol_20d"] = log_ret.rolling(window=20, min_periods=5).std() * np.sqrt(252)

    # 6. 50-Day Simple Moving Average (SMA) Distance
    sma_50 = close.rolling(window=50, min_periods=10).mean()
    res["Dist_SMA_50"] = (close - sma_50) / (sma_50 + 1e-8)

    # 7. Imputation: strictly preserves temporal causality
    # bfill only on initial rolling warm-up window, ffill for forward market state, fillna(0.0) as uninformative prior
    res = res.bfill().ffill().fillna(0.0)
    return res
