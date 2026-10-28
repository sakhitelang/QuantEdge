"""
QuantEdge AI — Reinforcement Learning Engine (Basic Actor-Critic)
Custom Markov Decision Process (MDP) for Indian Equities (NSE / BSE).
Deep Actor-Critic neural network with exact 9,573 parameter architecture,
Basic TD Actor-Critic policy gradient and value baseline estimation, semi-variance penalty, and Indian transaction friction modeling.
"""

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Any, Optional

# Action mapping to target equity allocation weights
ACTION_NAMES = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
    3: "STRONG_BUY",
}

ACTION_WEIGHTS = {
    0: 0.00,   # 100% Cash / INR
    1: 0.25,   # Defensive 25% Allocation
    2: 0.65,   # Moderate Conviction 65% Allocation
    3: 1.00,   # High Conviction 100% Bullish Allocation
}


# =====================================================================
# State Preprocessing & Normalization Pipeline
# =====================================================================

def prepare_market_state_matrix(
    df: pd.DataFrame,
    vix_val: float = 14.5,
    nifty_ret: float = 0.001,
) -> np.ndarray:
    """
    Transforms raw and engineered features into a zero-centered, bounded
    10-dimensional market observation matrix [T x 10].
    """
    close = df["Close"].values.astype(np.float32)
    t_len = len(df)

    ret_1d = np.clip(df["Ret_1d"].values.astype(np.float32) * 10.0, -1.0, 1.0)
    ret_5d = np.clip(df["Ret_5d"].values.astype(np.float32) * 5.0, -1.0, 1.0)
    ret_20d = np.clip(df["Ret_20d"].values.astype(np.float32) * 3.0, -1.0, 1.0)

    # RSI normalized to [-1, 1]
    rsi = df["RSI_14"].values.astype(np.float32)
    norm_rsi = np.clip((rsi - 50.0) / 50.0, -1.0, 1.0)

    # MACD Histogram scaled by close price, clamped to [-1, 1]
    macd_hist = df["MACD_hist"].values.astype(np.float32)
    norm_macd = np.clip((macd_hist / (close + 1e-5)) * 50.0, -1.0, 1.0)

    # ATR Ratio normalized by close price, clamped to [0, 1]
    atr = df["ATRr_14"].values.astype(np.float32)
    norm_atr = np.clip((atr / (close + 1e-5)) * 20.0, 0.0, 1.0)

    # Realized Volatility scaled against 50% annual threshold, clamped to [0, 1]
    vol = df["Realized_Vol_20d"].values.astype(np.float32)
    norm_vol = np.clip(vol / 0.5, 0.0, 1.0)

    # SMA Distance clamped to [-1, 1]
    dist_sma = df["Dist_SMA_50"].values.astype(np.float32)
    norm_sma = np.clip(dist_sma * 5.0, -1.0, 1.0)

    # Macro factors: INDIA VIX centered at baseline 15.0, NIFTY return scaled
    norm_vix = np.full(t_len, np.clip((vix_val - 15.0) / 15.0, -1.0, 1.0), dtype=np.float32)
    norm_nifty = np.full(t_len, np.clip(nifty_ret * 20.0, -1.0, 1.0), dtype=np.float32)

    mat = np.column_stack([
        ret_1d,
        ret_5d,
        ret_20d,
        norm_rsi,
        norm_macd,
        norm_atr,
        norm_vol,
        norm_sma,
        norm_vix,
        norm_nifty,
    ])

    mat = np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    return mat


# =====================================================================
# Indian Trading Environment (MDP Simulator)
# =====================================================================

class IndianTradingEnv:
    """
    Markov Decision Process (MDP) modeling Indian equity trading frictions:
    - State Space: 12-dimensional vector (10 market features + current_weight + unrealized_pnl)
    - Action Space: 4 discrete portfolio targets (SELL=0%, HOLD=25%, BUY=65%, STRONG_BUY=100%)
    - Reward: Step return penalized for 0.1% transaction friction, downside semi-variance, and peak drawdown.
    """

    def __init__(
        self,
        market_matrix: np.ndarray,
        asset_returns: np.ndarray,
        friction_rate: float = 0.001,
        downside_penalty_coef: float = 0.5,
        drawdown_penalty_coef: float = 0.2,
    ):
        self.market_matrix = market_matrix
        self.asset_returns = asset_returns
        self.n_steps = len(asset_returns)
        self.friction_rate = friction_rate
        self.downside_penalty_coef = downside_penalty_coef
        self.drawdown_penalty_coef = drawdown_penalty_coef

        self.current_step = 0
        self.current_weight = 0.0
        self.unrealized_pnl = 0.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.current_weight = 0.0
        self.unrealized_pnl = 0.0
        self.portfolio_value = 1.0
        self.peak_value = 1.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        step = min(self.current_step, self.n_steps - 1)
        m_feats = self.market_matrix[step]
        state = np.append(m_feats, [self.current_weight, self.unrealized_pnl]).astype(np.float32)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        target_weight = ACTION_WEIGHTS[action]
        asset_ret = float(self.asset_returns[self.current_step])

        # 1. Indian Statutory Friction Penalty (0.1% on turnover)
        weight_delta = abs(target_weight - self.current_weight)
        cost_penalty = self.friction_rate * weight_delta

        # 2. Step Portfolio Return
        step_return = (self.current_weight * asset_ret) - cost_penalty

        # 3. Position tracking & Unrealized PnL
        self.unrealized_pnl = (self.current_weight * asset_ret)
        self.current_weight = target_weight
        self.portfolio_value *= (1.0 + step_return)
        self.portfolio_value = max(1e-4, self.portfolio_value)

        # 4. Asymmetric Downside Semi-Variance Penalty
        vol_penalty = (self.downside_penalty_coef * (step_return ** 2)) if step_return < 0.0 else 0.0

        # 5. Peak Drawdown Penalty
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        dd_penalty = self.drawdown_penalty_coef * drawdown

        # Composite institutional reward
        reward = step_return - vol_penalty - dd_penalty

        self.current_step += 1
        done = self.current_step >= (self.n_steps - 1)
        next_state = self._get_state()

        info = {
            "portfolio_value": self.portfolio_value,
            "step_return": step_return,
            "drawdown": drawdown,
            "weight": self.current_weight,
        }
        return next_state, reward, done, info


# =====================================================================
# Deep Actor-Critic Neural Network Architecture (9,573 Parameters)
# =====================================================================

class ActorCriticNetwork(nn.Module):
    """
    Institutional Actor-Critic architecture with shared latent representation:
    - Input: 12-dim state vector
    - Backbone L1: Linear(12 -> 64), LayerNorm(64), LeakyReLU(0.1)  [832 + 128 = 960 params]
    - Backbone L2: Linear(64 -> 64), LayerNorm(64), LeakyReLU(0.1)  [4160 + 128 = 4288 params]
    - Actor Head:  Linear(64 -> 32), LeakyReLU(0.1), Linear(32 -> 4) [2080 + 132 = 2212 params]
    - Critic Head: Linear(64 -> 32), LeakyReLU(0.1), Linear(32 -> 1) [2080 + 33 = 2113 params]
    Total trainable parameters = 9,573.
    """

    def __init__(self, state_dim: int = 12, action_dim: int = 4):
        super(ActorCriticNetwork, self).__init__()

        # Shared Feature Extractor Backbone
        self.backbone_fc1 = nn.Linear(state_dim, 64)
        self.backbone_ln1 = nn.LayerNorm(64)
        self.act1 = nn.LeakyReLU(negative_slope=0.1)

        self.backbone_fc2 = nn.Linear(64, 64)
        self.backbone_ln2 = nn.LayerNorm(64)
        self.act2 = nn.LeakyReLU(negative_slope=0.1)

        # Specialized Actor Policy Head
        self.actor_fc = nn.Linear(64, 32)
        self.actor_act = nn.LeakyReLU(negative_slope=0.1)
        self.actor_out = nn.Linear(32, action_dim)

        # Specialized Critic Value Head
        self.critic_fc = nn.Linear(64, 32)
        self.critic_act = nn.LeakyReLU(negative_slope=0.1)
        self.critic_out = nn.Linear(32, 1)

        self._warm_start_weights()

    def _warm_start_weights(self):
        """Orthogonal weight initialization for stable gradient propagation on financial series."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action logits [batch, 4] and scalar state-value V(s) [batch, 1]."""
        x = self.act1(self.backbone_ln1(self.backbone_fc1(state)))
        h = self.act2(self.backbone_ln2(self.backbone_fc2(x)))

        # Actor head -> logits
        a_mid = self.actor_act(self.actor_fc(h))
        logits = self.actor_out(a_mid)

        # Critic head -> value V(s)
        c_mid = self.critic_act(self.critic_fc(h))
        val = self.critic_out(c_mid)

        return logits, val

    def get_action_and_value(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        Inference helper:
        - If deterministic: uses Argmax on Softmax distribution.
        - If stochastic: samples from Categorical distribution.
        Returns: (action, log_prob, value, action_probabilities)
        """
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = Categorical(probs=probs)

        if deterministic:
            action_idx = int(torch.argmax(probs, dim=-1).item())
        else:
            action_idx = int(dist.sample().item())

        log_prob = float(dist.log_prob(torch.tensor(action_idx)).item())
        val_scalar = float(value.squeeze().item())
        probs_np = probs.detach().cpu().numpy().squeeze()

        return action_idx, log_prob, val_scalar, probs_np
# =====================================================================
# Basic Actor-Critic Optimizer
# =====================================================================

class ActorCriticOptimizer:
    """
    Foundational Actor-Critic (AC) optimization engine:
    - Pure Actor-Critic architecture without surrogate clipping, policy ratios, or GAE.
    - 1-step TD Error (delta): delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    - Actor Policy Gradient Loss: -E[ log_prob(a) * delta ]
    - Critic Value Baseline MSE Loss: 0.5 * E[ (V(s) - Target)^2 ]
    - Standard optimizer update without extra heuristic modifications.
    """

    def __init__(
        self,
        network: ActorCriticNetwork,
        lr: float = 1.8e-3,
        gamma: float = 0.99,
        vf_coef: float = 0.5,
    ):
        self.network = network
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def train_rollout(
        self,
        env: IndianTradingEnv,
        epochs: int = 1,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Executes a full episode rollout and optimizes policy via basic Actor-Critic."""
        state = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []

        done = False
        while not done:
            st_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = self.network.get_action_and_value(st_tensor, deterministic=False)

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1.0 if done else 0.0)

            state = next_state

        b_states = torch.tensor(np.array(states), dtype=torch.float32)
        b_actions = torch.tensor(np.array(actions), dtype=torch.long)
        b_rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        b_next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        b_dones = torch.tensor(np.array(dones), dtype=torch.float32)

        # Forward pass on rollout states
        logits, val_pred = self.network(b_states)
        values = val_pred.squeeze(-1)

        # 1-step bootstrap target value: r + gamma * V(s') * (1 - done)
        with torch.no_grad():
            _, next_val_pred = self.network(b_next_states)
            next_values = next_val_pred.squeeze(-1)
            target_values = b_rewards + self.gamma * next_values * (1.0 - b_dones)

        # TD Error (delta)
        td_error = (target_values - values).detach()

        # Actor Policy Loss: -E[ log_prob(a) * delta ]
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(b_actions)
        policy_loss = -(log_probs * td_error).mean()

        # Critic MSE Baseline Loss: 0.5 * E[ (V(s) - Target)^2 ]
        val_loss = 0.5 * ((values - target_values) ** 2).mean()

        # Total Basic Actor-Critic Loss
        loss = policy_loss + (self.vf_coef * val_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(val_loss.item()),
            "final_portfolio_value": env.portfolio_value,
        }


# =====================================================================
# High-Level Reinforcement Learning Engine Wrapper
# =====================================================================

class RLForecastingSystem:
    """
    High-level orchestrator for training, backtesting, and generating institutional
    trading signals for Indian equities using local data files.
    """

    def __init__(self, state_dim: int = 12, action_dim: int = 4, lr: float = 1.8e-3):
        self.network = ActorCriticNetwork(state_dim=state_dim, action_dim=action_dim)
        self.optimizer = ActorCriticOptimizer(self.network, lr=lr)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)

    def train_on_dataframe(
        self,
        df: pd.DataFrame,
        vix_val: float = 14.5,
        nifty_ret: float = 0.001,
        epochs: int = 1,
        repeat_cycles: int = 3,
    ) -> List[Dict[str, float]]:
        """Trains the basic Actor-Critic agent on local historical market data."""
        mat = prepare_market_state_matrix(df, vix_val=vix_val, nifty_ret=nifty_ret)
        asset_rets = df["Ret_1d"].values.astype(np.float32)

        history = []
        for cycle in range(repeat_cycles):
            env = IndianTradingEnv(market_matrix=mat, asset_returns=asset_rets)
            stats = self.optimizer.train_rollout(env, epochs=epochs)
            stats["cycle"] = cycle + 1
            history.append(stats)

        return history

    def evaluate_strategy(
        self,
        df: pd.DataFrame,
        vix_val: float = 14.5,
        nifty_ret: float = 0.001,
    ) -> Dict[str, Any]:
        """
        Backtests the trained deterministic policy across the full dataset.
        Computes Sharpe ratio, Max Drawdown, CAGR, and Equity Curve.
        """
        mat = prepare_market_state_matrix(df, vix_val=vix_val, nifty_ret=nifty_ret)
        asset_rets = df["Ret_1d"].values.astype(np.float32)
        env = IndianTradingEnv(market_matrix=mat, asset_returns=asset_rets)

        state = env.reset()
        done = False
        trajectory = []

        self.network.eval()
        with torch.no_grad():
            while not done:
                st_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action, _, val, probs = self.network.get_action_and_value(st_t, deterministic=True)
                next_state, reward, done, info = env.step(action)

                trajectory.append({
                    "step": env.current_step,
                    "action": ACTION_NAMES[action],
                    "weight": ACTION_WEIGHTS[action],
                    "reward": reward,
                    "portfolio_value": info["portfolio_value"],
                    "probs": probs,
                    "critic_val": val,
                })
                state = next_state

        # Performance metrics
        port_values = np.array([t["portfolio_value"] for t in trajectory])
        step_returns = np.diff(port_values) / port_values[:-1] if len(port_values) > 1 else np.array([0.0])

        total_return = float(port_values[-1] - 1.0) if len(port_values) > 0 else 0.0
        n_days = len(port_values)
        ann_factor = 252.0 / max(1, n_days)
        cagr = float(((port_values[-1]) ** (ann_factor / 252.0)) - 1.0) if port_values[-1] > 0 else -1.0

        std_ret = float(np.std(step_returns))
        sharpe = float((np.mean(step_returns) / (std_ret + 1e-8)) * np.sqrt(252)) if std_ret > 0 else 0.0

        # Max drawdown
        running_max = np.maximum.accumulate(port_values)
        drawdowns = (running_max - port_values) / (running_max + 1e-8)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "final_value": float(port_values[-1]),
            "trajectory_length": len(trajectory),
            "trajectory": trajectory,
        }

    def predict_current_signal(
        self,
        df: pd.DataFrame,
        vix_val: float = 14.5,
        nifty_ret: float = 0.001,
        current_holding_weight: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Executes real-time inference on the latest market state row.
        Outputs greedy action, Softmax probability distribution, confidence score, and Critic valuation.
        """
        mat = prepare_market_state_matrix(df, vix_val=vix_val, nifty_ret=nifty_ret)
        latest_market = mat[-1]
        latest_state = np.append(latest_market, [current_holding_weight, 0.0]).astype(np.float32)

        self.network.eval()
        with torch.no_grad():
            st_tensor = torch.tensor(latest_state, dtype=torch.float32).unsqueeze(0)
            logits, val = self.network(st_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            action_idx = int(np.argmax(probs))

        action_name = ACTION_NAMES[action_idx]
        confidence = float(probs[action_idx])
        target_weight = ACTION_WEIGHTS[action_idx]
        critic_val = float(val.squeeze().item())

        prob_breakdown = {ACTION_NAMES[i]: float(probs[i]) for i in range(4)}

        return {
            "action": action_name,
            "action_id": action_idx,
            "target_weight": target_weight,
            "confidence": confidence,
            "critic_value": critic_val,
            "probability_distribution": prob_breakdown,
        }
