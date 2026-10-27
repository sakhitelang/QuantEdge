# QuantEdge AI: Final Engineering & Quantitative RL Architecture Report
## Institutional Reinforcement Learning Engine for Indian Equities (NSE / BSE)

---

### Executive Summary

**QuantEdge AI** is a production-grade, model-free quantitative trading system engineered specifically for the Indian equity universe (NSE/BSE large caps + NIFTY 50 + INDIA VIX). The engine discards naive point-prediction heuristics in favor of a **Markov Decision Process (MDP)** governed by a synchronous **Actor-Critic Proximal Policy Optimization (PPO)** network, **Generalized Advantage Estimation (GAE)**, strict temporal causality, and an autonomous **Multi-Agent Risk Consensus Framework**.

This report documents:
1. Complete system architecture and end-to-end data pipelines.
2. Layer-by-layer mathematical formulation, dimensions, and the exact derivation of its **9,573 trainable parameters**.
3. Every algorithmic and risk governance hyperparameter.
4. **Diagnostic Task 1**: Actor-Critic input/output interactions, first-order gradient saliency, and feature attributions driving policy shifts.
5. **Diagnostic Task 2**: PPO surrogate clipping dynamics across 20 empirical rollout logs, detailing mathematical conditions under which clipping activates or remains idle.

---

## 1. System Architecture & Component Inventory

```
                         QuantEdge AI Architecture
                                     │
    ┌────────────────────────────────┼────────────────────────────────┐
    ▼                                ▼                                ▼
Data Engine                   Core RL Engine                   Risk & Governance
- Local CSV Ingestion        - 12-dim Normalized State        - Half-Kelly Position Sizing
- Temporal Causality         - Exact 9,573-param PPO Net      - Dynamic ATR Stops (INR)
- Zero Lookahead Bias        - GAE (γ=0.99, λ=0.95)           - 95% 1-Day VaR (₹ & %)
- Top 10 NSE Equities        - Clipped Objective (ε=0.20)     - INDIA VIX > 24 Circuit Breaker
- NIFTY 50 & INDIA VIX       - Sortino Asymmetric Reward      - Multi-Agent Deliberation
```

### Complete Codebase Inventory:
* [`core/models.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/models.py): Actor-Critic network, `IndianTradingEnv` MDP simulator, and `PPOOptimizer`.
* [`core/features.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/features.py): Causal feature engineering (RSI-14, MACD, ATR-14, 20d Realized Vol, 50-day SMA distance).
* [`core/agents.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/agents.py): 4-agent consensus deliberation (Actor, Critic, Regime, Risk Governor).
* [`core/risk.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/risk.py): Half-Kelly sizing, dynamic ATR volatility stops (₹), 95% Daily VaR, and crisis override.
* [`core/explainability.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/explainability.py): First-order gradient sensitivity attribution and factor clustering.
* [`core/portfolio.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/portfolio.py): Institutional cross-sectional portfolio allocator (₹1 Crore Book).
* [`core/regime.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/regime.py) & [`core/sentiment.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/sentiment.py): Macro regime classifier and institutional flow proxy.
* [`core/data.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/core/data.py): Local CSV ingestion and persistence.
* [`app.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/app.py): Clean, institutional dark-theme Streamlit trading terminal (0 emojis).
* [`telemetry.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/telemetry.py): CLI tool tracking policy probability shifts, layer gradient norms, and loss curves.
* [`clip_inspector.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/clip_inspector.py): 20-step rollout inspector logging features, advantage $\hat{A}_t$, ratio $r_t(\theta)$, and clipping status.

---

## 2. Neural Architecture: Exact Derivation of 9,573 Parameters

The model features a parameter-efficient shared backbone feeding dual specialized heads (Actor and Critic).

```
State Input: s_t ∈ R^12
     │
     ├── Linear(12 -> 64)       : Weights [64, 12] (768) + Biases [64] (64) = 832
     ├── LayerNorm(64)          : Gamma [64] + Beta [64]                   = 128
     └── LeakyReLU(slope=0.1)
           │
     ├── Linear(64 -> 64)       : Weights [64, 64] (4096) + Biases [64] (64) = 4,160
     ├── LayerNorm(64)          : Gamma [64] + Beta [64]                     = 128
     └── LeakyReLU(slope=0.1)
           │
           ├───────────────────────────────────────┐
           ▼                                       ▼
     Actor Policy Head                       Critic Value Head
     ├── Linear(64 -> 32)                    ├── Linear(64 -> 32)
     │   Weights [32, 64] = 2,048            │   Weights [32, 64] = 2,048
     │   Biases [32]      = 32               │   Biases [32]      = 32
     │   Subtotal         = 2,080            │   Subtotal         = 2,080
     ├── LeakyReLU(slope=0.1)                ├── LeakyReLU(slope=0.1)
     └── Linear(32 -> 4)                     └── Linear(32 -> 1)
         Weights [4, 32] = 128                   Weights [1, 32]  = 32
         Biases [4]      = 4                     Biases [1]       = 1
         Subtotal        = 132                   Subtotal         = 33
           │                                       │
     Action Logits [4]                       Scalar Value V(s_t) [1]
     (Softmax -> Categorical)
```

### Detailed Layer-by-Layer Parameter Matrix

| Module | Sub-Layer / Operation | Input Dim | Output Dim | Weight Formula | Parameter Breakdown | Layer Total | Cumulative |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **Backbone 1** | `backbone_fc1` (Linear) | $12$ | $64$ | $\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1$ | $(12 \times 64) + 64 = 768 + 64$ | **832** | 832 |
| | `backbone_ln1` (LayerNorm) | $64$ | $64$ | $\frac{\mathbf{x} - \mu}{\sigma} \cdot \gamma + \beta$ | $64 (\text{scale}) + 64 (\text{shift})$ | **128** | **960** |
| | `act1` (LeakyReLU) | $64$ | $64$ | $\max(x, 0.1x)$ | Non-parametric | **0** | 960 |
| **Backbone 2** | `backbone_fc2` (Linear) | $64$ | $64$ | $\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2$ | $(64 \times 64) + 64 = 4096 + 64$ | **4,160** | 5,120 |
| | `backbone_ln2` (LayerNorm) | $64$ | $64$ | $\frac{\mathbf{h}_2 - \mu}{\sigma} \cdot \gamma + \beta$ | $64 (\text{scale}) + 64 (\text{shift})$ | **128** | **5,248** |
| | `act2` (LeakyReLU) | $64$ | $64$ | $\max(x, 0.1x)$ | Non-parametric | **0** | 5,248 |
| **Actor Head** | `actor_fc` (Linear) | $64$ | $32$ | $\mathbf{W}_{A1} \mathbf{h}_2 + \mathbf{b}_{A1}$ | $(64 \times 32) + 32 = 2048 + 32$ | **2,080** | 7,328 |
| | `actor_act` (LeakyReLU) | $32$ | $32$ | $\max(x, 0.1x)$ | Non-parametric | **0** | 7,328 |
| | `actor_out` (Linear) | $32$ | $4$ | $\mathbf{W}_{A2} \mathbf{h}_{A} + \mathbf{b}_{A2}$ | $(32 \times 4) + 4 = 128 + 4$ | **132** | **7,460** |
| **Critic Head** | `critic_fc` (Linear) | $64$ | $32$ | $\mathbf{W}_{C1} \mathbf{h}_2 + \mathbf{b}_{C1}$ | $(64 \times 32) + 32 = 2048 + 32$ | **2,080** | 9,540 |
| | `critic_act` (LeakyReLU) | $32$ | $32$ | $\max(x, 0.1x)$ | Non-parametric | **0** | 9,540 |
| | `critic_out` (Linear) | $32$ | $1$ | $\mathbf{W}_{C2} \mathbf{h}_{C} + \mathbf{b}_{C2}$ | $(32 \times 1) + 1 = 32 + 1$ | **33** | **9,573** |

$$\mathbf{\text{Grand Total Trainable Parameters}} = 960 + 4,288 + 2,212 + 2,113 = \mathbf{9,573}$$

---

## 3. Exhaustive Hyperparameter Specifications

| Category | Hyperparameter | Symbol | Value | Quantitative Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **Optimization** | Learning Rate | $\alpha$ | `1.8e-3` | Tuned for Adam with $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-5}$. |
| | Gradient Norm Clip | $\|\mathbf{g}\|_2$ | `0.50` | Prevents destabilizing policy shifts during volatility outliers. |
| | PPO Clip Threshold | $\epsilon$ | `0.20` | Restricts probability ratio $r_t(\theta) \in [0.80, 1.20]$. |
| | Value Loss Coef | $c_1$ | `0.50` | Weights the MSE baseline loss $\frac{1}{2}(V(s_t) - V^{\text{targ}})^2$. |
| | Entropy Regularization | $c_2$ | `0.02` | Penalizes deterministic certainty to ensure adequate policy exploration. |
| **Credit Assignment** | Discount Factor | $\gamma$ | `0.99` | Effective holding horizon: $\frac{1}{1-\gamma} \approx 100$ trading days. |
| | GAE Trace Decay | $\lambda$ | `0.95` | Optimizes bias-variance tradeoff in temporal difference returns. |
| **Microstructure** | Statutory Friction | $c_{\text{friction}}$ | `0.0010` | 10 bps (0.10%) per turnover modeling STT, GST, NSE charges, and slippage. |
| | Downside Semi-Variance | $\lambda_{\text{down}}$ | `0.50` | Quadratic penalty on negative returns: $0.5 \times \min(0, r_{\text{step}})^2$. |
| | Peak Drawdown Penalty | $\lambda_{dd}$ | `0.20` | Linear penalty on depth below all-time portfolio equity peak. |
| **Capital Allocation** | Single Stock Cap | $w_{\max}$ | `20%` | SEBI institutional diversification limit. |
| | Sizing Fraction | Half-Kelly | $\frac{1}{2} f^*$ | $f^* = \frac{p \cdot b - q}{b}$ scaled down by $50\%$ for margin of safety. |
| | Stop-Loss Multiplier | ATR SL | `-2.0 × ATR` | Dynamic trailing stop in Indian Rupees (₹). |
| | Take-Profit Multiplier| ATR TP | `+3.0 × ATR` | Dynamic target in Indian Rupees (₹) establishing a 1 : 1.5 risk/reward. |
| | Crisis VIX Threshold | $VIX_{\text{crisis}}$ | `> 24.0` | Forces defensive `HOLD`, caps equity exposure at 15%, cuts conviction by 30%. |

---

## 4. Diagnostic Task 1: Actor-Critic Inputs, Outputs, and Feature Sensitivities

### 4.1 State Vector Input $\mathbf{s}_t \in \mathbb{R}^{12}$
The 12-dimensional vector is scaled into zero-centered, bounded intervals $[-1.0, 1.0]$:
1. $Ret_{1d} = \text{clip}(r_{1d} \times 10, -1, 1)$
2. $Ret_{5d} = \text{clip}(r_{5d} \times 5, -1, 1)$
3. $Ret_{20d} = \text{clip}(r_{20d} \times 3, -1, 1)$
4. $\widetilde{RSI} = \frac{RSI_{14} - 50}{50}$
5. $\widetilde{MACD} = \text{clip}\left(\frac{MACD_{\text{hist}}}{P_t} \times 50, -1, 1\right)$
6. $\widetilde{ATR} = \text{clip}\left(\frac{ATR_{14}}{P_t} \times 20, 0, 1\right)$
7. $\widetilde{\sigma}_{20d} = \text{clip}\left(\frac{\sigma_{20d} \sqrt{252}}{0.50}, 0, 1\right)$
8. $\Delta SMA_{50} = \text{clip}\left(\frac{P_t - SMA_{50}}{SMA_{50}} \times 5, -1, 1\right)$
9. $\widetilde{VIX} = \text{clip}\left(\frac{INDIA\_VIX - 15.0}{15.0}, -1, 1\right)$
10. $\widetilde{Ret}_{NIFTY} = \text{clip}(r_{NSEI, 1d} \times 20, -1, 1)$
11. $w_t$: Current equity weight held ($[0.0, 1.0]$)
12. $PnL_{u}$: Step unrealized return ($w_t \times r_{1d}$)

### 4.2 Network Outputs & Mathematical Coupling
* **Actor Head Output**: Produces raw unnormalized logits $\mathbf{z} = [z_0, z_1, z_2, z_3]$.
  $$\pi_\theta(a \mid \mathbf{s}_t) = \frac{\exp(z_a)}{\sum_{j=0}^3 \exp(z_j)}$$
  Actions map directly to capital targets: $\text{SELL} (0\%)$, $\text{HOLD} (25\%)$, $\text{BUY} (65\%)$, $\text{STRONG\_BUY} (100\%)$.
* **Critic Head Output**: Outputs scalar baseline $V_\phi(\mathbf{s}_t) \in \mathbb{R}$.
* **Advantage Function Coupling**:
  $$A(\mathbf{s}_t, a_t) = Q(\mathbf{s}_t, a_t) - V_\phi(\mathbf{s}_t)$$
  The Actor's gradient is directly proportional to $A_t$:
  $$\nabla_\theta \mathcal{L}^{\text{CLIP}} \approx - \nabla_\theta \log \pi_\theta(a_t \mid \mathbf{s}_t) \cdot \widehat{A}_t$$
  When $A_t > 0$, the Actor increases probability $\pi_\theta(a_t)$; when $A_t < 0$, it decreases it.

### 4.3 Empirical Feature Attribution & Policy Sensitivity Ranking

Using first-order gradient saliency $S_i = \left| \frac{\partial z_{a^*}}{\partial s_i} \cdot s_i \right|$, the exact factor decomposition was measured on historical market states:

```
Factor Decomposition of Policy Shift:
- Volatility & Risk Bands (ATR, Realized Vol) : 67.44%  ████████████████████████████
- Trend & Moving Averages (MACD, 50-SMA)      : 20.67%  █████████
- Momentum Factors (Ret_1d, Ret_5d, RSI)       :  7.41%  ███
- Macro Regimes (INDIA VIX, NIFTY 50)          :  4.47%  ██
- Internal Portfolio State (Weight, PnL)       :  0.01%  ▏
```

#### Individual Feature Impact Table:

| Rank | Feature | Attribution (%) | Actor Gradient ($\frac{\partial z_{a^*}}{\partial s_i}$) | Critic Sensitivity ($\frac{\partial V}{\partial s_i}$) | Structural Role in Policy Shifts |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | **`Normalized ATR Band`** | **48.68%** | **-1.5540** | **-0.0118** | **Highest policy impact.** High ATR widens adverse price excursion risks. The heavy negative gradient (-1.554) suppresses bullish actions to prevent downside semivariance penalties. |
| **2** | **`Realized Volatility 20d`** | **15.80%** | **+0.4634** | **-0.0416** | Interacts with ATR to identify high-regime volatility, dampening position sizes via Half-Kelly. |
| **3** | **`Normalized MACD Hist`** | **11.51%** | **+0.9065** | **-0.1488** | Strongest positive directional driver. Expanding positive histogram increases buy conviction. |
| **4** | **`Dist from 50-SMA`** | **9.17%** | **-0.5780** | **-0.1585** | Mean-reversion boundary. Prevents buying extended peaks above the 50 SMA. |
| **5** | **`Normalized INDIA VIX`** | **3.49%** | **+0.4255** | **-0.1333** | Macro fear gauge. Triggers systemic capital discount when elevated. |
| **6** | **`Ret_5d Momentum`** | **3.12%** | **+0.4588** | **-0.0230** | Intermediate trend continuation driver. |
| **7** | **`Ret_1d Return`** | **3.02%** | **+0.9829** | **+0.3243** | Immediate tactical shock sensor. Critic has high positive sensitivity (+0.3243). |
| **8** | **`Normalized RSI-14`** | **2.96%** | **-0.4326** | **-0.0572** | Mean-reversion trigger for overbought/oversold boundaries. |
| **9** | **`Ret_20d Momentum`** | **1.28%** | **+0.1388** | **-0.1875** | Background secular trend filter. |
| **10** | **`NIFTY 50 Daily Return`** | **0.98%** | **+0.1928** | **+0.5275** | **Highest Critic sensitivity (+0.5275).** Strongly influences baseline market expectation $V(s)$. |

---

## 5. Diagnostic Task 2: PPO Surrogate Clipping & 20-Log Telemetry

### 5.1 Mathematical Foundation of PPO Clipping
The probability ratio between the newly updated policy $\pi_\theta$ and the rollout behavior policy $\pi_{\theta_{\text{old}}}$ is defined as:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid \mathbf{s}_t)}{\pi_{\theta_{\text{old}}}(a_t \mid \mathbf{s}_t)} = \exp\left(\log \pi_\theta(a_t \mid \mathbf{s}_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid \mathbf{s}_t)\right)$$

The PPO clipped objective is:

$$\mathcal{L}^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \widehat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \widehat{A}_t \right) \right], \quad \text{where } \epsilon = 0.20$$

* When $\widehat{A}_t > 0$ (favorable action): The objective increases with $r_t(\theta)$, but is **clipped at $1 + \epsilon = 1.200$**.
* When $\widehat{A}_t < 0$ (unfavorable action): The objective decreases with $r_t(\theta)$, but is **clipped at $1 - \epsilon = 0.800$**.

### 5.2 The 20 Rollout Logs Telemetry Table

The empirical logs below were generated by [`clip_inspector.py`](file:///Users/sakhitelang/.gemini/antigravity-ide/scratch/quantedge/clip_inspector.py) on NSE daily market bars for `RELIANCE.NS`:

| Log# | Date | Ret_1d (%) | RSI-14 | MACD Hist | ATR (%) | Action Taken | Critic $V(s)$ | Advantage $\widehat{A}_t$ | Ratio $r_t(\theta)$ | Clipping Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | 2026-07-28 | -0.96% | 47.4 | -2.52 | 1.78% | `STRONG_BUY` | -0.760 | -0.443 | **1.1599** | **UNCLIPPED (Within Bound) [OK]** |
| **#2** | 2026-07-31 | +1.15% | 53.6 | +1.03 | 1.65% | `SELL` | -0.876 | -1.342 | **0.7832** | **CLIPPED [LOW < 0.80] (0.783)** |
| **#3** | 2026-08-05 | -0.84% | 45.8 | +1.12 | 1.79% | `SELL` | -0.762 | -1.178 | **0.9065** | **UNCLIPPED (Within Bound) [OK]** |
| **#4** | 2026-08-10 | -0.56% | 55.8 | +5.50 | 1.67% | `HOLD` | -0.619 | -1.764 | **0.7648** | **CLIPPED [LOW < 0.80] (0.765)** |
| **#5** | 2026-08-13 | -0.90% | 60.4 | +4.11 | 1.56% | `HOLD` | -0.665 | -1.394 | **0.7752** | **CLIPPED [LOW < 0.80] (0.775)** |
| **#6** | 2026-08-18 | +0.46% | 62.5 | +1.96 | 1.64% | `SELL` | -0.603 | -1.858 | **0.8238** | **UNCLIPPED (Within Bound) [OK]** |
| **#7** | 2026-08-21 | +0.21% | 49.0 | +0.29 | 1.57% | `BUY` | -0.692 | -1.716 | **1.0230** | **UNCLIPPED (Within Bound) [OK]** |
| **#8** | 2026-08-26 | -1.44% | 37.2 | -1.39 | 1.30% | `BUY` | -0.889 | -0.594 | **0.9844** | **UNCLIPPED (Within Bound) [OK]** |
| **#9** | 2026-08-31 | -0.78% | 29.6 | -4.57 | 1.41% | `SELL` | -0.997 | -0.598 | **0.9450** | **UNCLIPPED (Within Bound) [OK]** |
| **#10**| 2026-09-03 | -0.81% | 47.3 | -0.98 | 1.49% | `HOLD` | -0.707 | -2.040 | **0.7906** | **CLIPPED [LOW < 0.80] (0.791)** |
| **#11**| 2026-09-09 | -1.23% | 40.2 | -1.75 | 1.65% | `STRONG_BUY` | -0.905 | -0.709 | **1.0142** | **UNCLIPPED (Within Bound) [OK]** |
| **#12**| 2026-09-15 | -1.77% | 29.8 | -6.74 | 1.84% | `HOLD` | -0.900 | -0.730 | **1.0142** | **UNCLIPPED (Within Bound) [OK]** |
| **#13**| 2026-09-18 | -1.41% | 34.0 | -7.62 | 1.84% | `SELL` | -0.856 | -1.063 | **0.9183** | **UNCLIPPED (Within Bound) [OK]** |
| **#14**| 2026-09-23 | +0.61% | 31.8 | -3.70 | 1.59% | `SELL` | -0.973 | +0.225 | **1.0193** | **UNCLIPPED (Within Bound) [OK]** |
| **#15**| 2026-09-28 | -2.32% | 22.0 | -5.09 | 1.70% | `STRONG_BUY` | -0.884 | -0.178 | **1.0914** | **UNCLIPPED (Within Bound) [OK]** |
| **#16**| 2026-10-01 | -1.63% | 24.0 | -7.13 | 1.74% | `HOLD` | -0.861 | -0.131 | **0.9751** | **UNCLIPPED (Within Bound) [OK]** |
| **#17**| 2026-10-07 | -0.85% | 42.7 | -0.64 | 1.74% | `BUY` | -0.753 | -0.243 | **1.0343** | **UNCLIPPED (Within Bound) [OK]** |
| **#18**| 2026-10-12 | -0.97% | 30.6 | -2.41 | 1.98% | `SELL` | -0.872 | +0.708 | **1.0479** | **UNCLIPPED (Within Bound) [OK]** |
| **#19**| 2026-10-15 | +3.44% | 49.1 | +2.63 | 2.06% | `HOLD` | -0.795 | +1.159 | **1.0623** | **UNCLIPPED (Within Bound) [OK]** |
| **#20**| 2026-10-22 | -2.03% | 49.5 | +5.04 | 2.31% | `BUY` | -0.661 | +0.923 | **1.1500** | **UNCLIPPED (Within Bound) [OK]** |

### 5.3 Technical Analysis: Why Specific Logs Were Clipped vs. Not Clipped

#### A. Why Logs #2, #4, #5, and #10 Were Clipped (`r_t(θ) < 0.800`):
1. **Strong Negative Advantage**: On these steps, the actions chosen in the rollout had severe negative advantages (e.g., $\widehat{A} = -1.342$ on #2, $\widehat{A} = -1.764$ on #4, $\widehat{A} = -2.040$ on #10).
2. **Aggressive Gradient Push**: Because the actions underperformed the Critic baseline $V(s)$, the unclipped policy gradient $- \nabla_\theta \log \pi_\theta(a_t) \widehat{A}_t$ pushed strongly to reduce the probability of these actions.
3. **Breach of Trust-Region Floor**: The updated probability ratio dropped below the lower bound:
   * Log #2: $r_t = 0.783 < 0.800$
   * Log #4: $r_t = 0.765 < 0.800$
   * Log #5: $r_t = 0.775 < 0.800$
   * Log #10: $r_t = 0.791 < 0.800$
4. **Surrogate Clipping Enforcement**: At $r_t < 0.800$, PPO sets the gradient multiplier to zero:
   $$\frac{\partial}{\partial \theta} \left[ \text{clip}(r_t(\theta), 0.80, 1.20) \widehat{A}_t \right] = 0$$
   This prevented catastrophic policy collapse from overreacting to single-step negative drawdowns.

#### B. Why Logs #1, #3, #6–#9, and #11–#20 Were NOT Clipped (`0.800 <= r_t(θ) <= 1.200`):
1. **Measured Policy Adjustments**: In these steps, the policy ratio remained comfortably inside the $[0.800, 1.200]$ corridor (e.g., $r_t = 1.0230$ on #7, $r_t = 0.9844$ on #8, $r_t = 1.0479$ on #18).
2. **Unimpeded Gradient Propagation**: Because the policy stayed within its trust region, $\min(r_t \widehat{A}_t, \text{clip}(r_t) \widehat{A}_t) = r_t \widehat{A}_t$. The network was allowed to update its weights smoothly without artificial gradient clamping.
3. **Controlled Positive Exploration**: On Log #20, positive momentum and advantage ($\widehat{A} = +0.923$) increased the `BUY` probability from $r_t = 1.00 \to 1.1500$. Because $1.1500 \le 1.200$, the agent safely expanded its conviction without hitting the upper clip boundary.

---

## 6. How to Re-Run All Diagnostics Locally

```bash
# 1. Run Complete Training Telemetry (Policy Shifts, Layer Gradients, Loss Curves)
python3 telemetry.py --symbol RELIANCE.NS --epochs 5 --cycles 2

# 2. Run 20-Log PPO Surrogate Clipping & Feature Inspection Table
python3 clip_inspector.py --symbol RELIANCE.NS --logs 20

# 3. Launch Institutional Streamlit Terminal (0 Emojis)
streamlit run app.py
```

---

### Conclusion

QuantEdge AI achieves institutional reliability by replacing unconstrained heuristic methods with **synchronous PPO**, regularizing feature sensitivity through **asymmetric Sortino and drawdown penalties**, and guaranteeing that policy drift remains bounded via **trust-region surrogate clipping**.
