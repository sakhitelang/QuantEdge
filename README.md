# QuantEdge AI — Indian Stock Market (NSE / BSE) Quantitative RL Engine

**QuantEdge AI** is an institutional-grade quantitative trading framework engineered specifically for the **National Stock Exchange of India (NSE)** and **Bombay Stock Exchange (BSE)**. The platform discards classical black-box heuristic predictors in favor of a **Pure Reinforcement Learning (RL) Framework** based on a custom **Markov Decision Process (MDP)**, deep **Actor-Critic Proximal Policy Optimization (PPO)** neural architecture, multi-agent consensus deliberation, and strict SEBI-compliant risk governance.

Market data is downloaded directly from Yahoo Finance to local CSV files in `data/`, and all feature computations, environment rollouts, backtests, and inference run directly on the local files.

---

## 1. Directory Structure

```
quantedge/
├── data/                               # Downloaded local CSV datasets (EOD daily bars)
│   ├── RELIANCE_NS_1y_1d.csv
│   ├── TCS_NS_1y_1d.csv
│   ├── HDFCBANK_NS_1y_1d.csv
│   ├── INFY_NS_1y_1d.csv
│   ├── ICICIBANK_NS_1y_1d.csv
│   ├── BHARTIARTL_NS_1y_1d.csv
│   ├── ITC_NS_1y_1d.csv
│   ├── SBIN_NS_1y_1d.csv
│   ├── LT_NS_1y_1d.csv
│   ├── TATAMOTORS_NS_1y_1d.csv
│   ├── INDEX_NSEI_1y_1d.csv           # Benchmark NIFTY 50
│   └── INDEX_INDIAVIX_1y_1d.csv       # Benchmark INDIA VIX
├── core/
│   ├── __init__.py                    # Core exports
│   ├── data.py                        # Yahoo Finance downloader, local CSV storage & loaders
│   ├── features.py                    # Temporal causality feature engineering (RSI, MACD, ATR, Vol, SMA)
│   ├── models.py                      # ActorCriticNetwork (9,573 params), IndianTradingEnv, PPOOptimizer
│   ├── regime.py                      # Indian Macro Regime Classifier (VIX & NIFTY momentum)
│   ├── risk.py                        # Half-Kelly sizing, ATR volatility stops (INR), 95% Daily VaR, Crisis override
│   ├── agents.py                      # Multi-agent consensus (Policy Actor, Value Critic, Regime, Risk Governor)
│   ├── explainability.py              # Gradient sensitivity attribution & factor decomposition
│   ├── sentiment.py                   # Indian market institutional sentiment (FII/DII proxy)
│   └── portfolio.py                   # 10-stock institutional portfolio allocator & simulator
├── main.py                            # CLI execution runner
└── requirements.txt                   # Dependencies (torch, pandas, numpy, yfinance)
```

---

## 2. Key Architecture & Quantitative Details

1. **Local Data Persistence**:
   - `core/data.py` downloads full 1-year OHLCV daily bars from Yahoo Finance for top 10 NSE large caps + `^NSEI` + `^INDIAVIX` into `data/*.csv`.
   - All subsequent training, backtesting, and live inference read directly from local files.
2. **Features with Strict Temporal Causality**:
   - 14-period RSI, MACD(12, 26, 9), 14-period ATR, 20-day annualized realized volatility, 50-day SMA distance, multi-horizon returns ($Ret_{1d}, Ret_{5d}, Ret_{20d}$).
   - Missing-value handling via `.bfill().ffill().fillna(0.0)` strictly eliminates lookahead bias.
3. **Deep Actor-Critic Architecture (Exact 9,573 Parameters)**:
   - Shared backbone: Linear(12 $\to$ 64) $\to$ LayerNorm(64) $\to$ LeakyReLU(0.1) $\to$ Linear(64 $\to$ 64) $\to$ LayerNorm(64) $\to$ LeakyReLU(0.1).
   - Actor Policy Head: Linear(64 $\to$ 32) $\to$ LeakyReLU(0.1) $\to$ Linear(32 $\to$ 4 logits).
   - Critic Value Head: Linear(64 $\to$ 32) $\to$ LeakyReLU(0.1) $\to$ Linear(32 $\to$ 1 scalar baseline).
4. **Action Space & Transaction Frictions**:
   - 4 discrete targets: `SELL` (0%), `HOLD` (25%), `BUY` (65%), `STRONG_BUY` (100%).
   - Indian market statutory turnover friction ($0.1\%$ per trade turnover).
   - Downside semi-variance penalty ($\lambda = 0.5$) and peak drawdown penalty ($0.2 \times \text{DD}$).
5. **Optimization**:
   - GAE ($\gamma = 0.99, \lambda = 0.95$), clipped surrogate objective ($\epsilon = 0.2$), entropy bonus ($0.02$), gradient clipping ($0.5$).
6. **Risk Governance & Capital Allocation**:
   - Half-Kelly sizing bounded by a $20\%$ hard single-stock cap.
   - Dynamic ATR volatility trailing stops and profit targets in Indian Rupees (₹).
   - 95% Daily Value at Risk (Historical & Parametric VaR).
   - Crisis regime interlock (INDIA VIX > 24 forces `HOLD`, caps exposure to 15%, discounts confidence by 30%).

---

## 3. Usage & CLI Commands

### Download All Data from Yahoo Finance to Local CSVs:
```bash
python3 main.py --download-only
```

### Analyze & Train RL Engine on a Specific Stock:
```bash
python3 main.py --symbol RELIANCE.NS --epochs 15
```
Options:
- `--symbol <TICKER>`: E.g., `RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`, `ITC.NS`
- `--epochs <N>`: Training epochs per PPO rollout cycle
- `--download`: Download/refresh latest data before executing

### Run Complete Multi-Asset Institutional Portfolio Simulation (10 Stocks):
```bash
python3 main.py --portfolio --epochs 10
```

### Launch Institutional Streamlit Web Terminal:
```bash
streamlit run app.py
```

### Run PPO Policy Telemetry (Loss, Gradients, Policy Shift):
```bash
python3 telemetry.py --symbol RELIANCE.NS --epochs 5 --cycles 2
```

### Inspect 20-Step PPO Surrogate Clipping & Features:
```bash
python3 clip_inspector.py --symbol RELIANCE.NS --logs 20
```
