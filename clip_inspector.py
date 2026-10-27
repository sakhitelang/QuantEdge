#!/usr/bin/env python3
"""
QuantEdge AI — PPO Clipped Surrogate & Feature Inspection Telemetry
Inspects 20 trading rollout steps, showing input features, Actor-Critic outputs,
PPO probability ratios r_t(theta), and whether PPO clipping (epsilon=0.20) was activated.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.data import load_ticker_data, get_macro_context, normalize_ticker
from core.features import compute_features
from core.models import (
    RLForecastingSystem,
    prepare_market_state_matrix,
    IndianTradingEnv,
    ACTION_NAMES,
)


def inspect_clipping(symbol: str = "RELIANCE.NS", num_logs: int = 20, clip_eps: float = 0.20):
    clean_sym = normalize_ticker(symbol)
    print("=" * 115)
    print(f"  PPO STEP-BY-STEP FEATURE & CLIPPING TELEMETRY ({num_logs} LOGS): {clean_sym}")
    print(f"  Surrogate Trust-Region Bounds: [1 - ε, 1 + ε] = [{1.0 - clip_eps:.2f}, {1.0 + clip_eps:.2f}] (ε = {clip_eps})")
    print("=" * 115)

    # 1. Load Data
    raw_df = load_ticker_data(clean_sym)
    feat_df = compute_features(raw_df)
    vix_val, nifty_ret = get_macro_context()

    mat = prepare_market_state_matrix(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
    asset_rets = feat_df["Ret_1d"].values.astype(np.float32)

    # 2. Train baseline network for 1 cycle to establish historical policy
    rl_sys = RLForecastingSystem()
    net = rl_sys.network
    env = IndianTradingEnv(mat, asset_rets)

    # Rollout collection
    state = env.reset()
    states, actions, rewards, values, old_log_probs, dates = [], [], [], [], [], []
    done = False
    step_idx = 0

    while not done:
        st_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act, lp, val, _ = net.get_action_and_value(st_t, deterministic=False)
        nxt, rew, done, _ = env.step(act)

        states.append(state)
        actions.append(act)
        rewards.append(rew)
        values.append(val)
        old_log_probs.append(lp)
        
        # Match with dataframe timestamp
        ts = feat_df.index[min(step_idx, len(feat_df) - 1)].strftime("%Y-%m-%d")
        dates.append(ts)

        state = nxt
        step_idx += 1

    # Compute terminal value and GAE Advantages
    st_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        _, last_v = net(st_t)
        values.append(float(last_v.squeeze().item()))

    t_steps = len(rewards)
    advantages = np.zeros(t_steps, dtype=np.float32)
    gae = 0.0
    gamma = 0.99
    gae_lambda = 0.95

    for t in reversed(range(t_steps)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae

    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 3. Simulate a gradient update step to perturb the policy and observe clipping
    b_states = torch.tensor(np.array(states), dtype=torch.float32)
    b_actions = torch.tensor(np.array(actions), dtype=torch.long)
    b_old_lp = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
    b_adv = torch.tensor(norm_advantages, dtype=torch.float32)

    # Perform a few optimization steps so policy shifts
    optimizer = rl_sys.optimizer.optimizer
    for _ in range(4):
        logits, val_p = net(b_states)
        dist = Categorical(logits=logits)
        cur_lp = dist.log_prob(b_actions)
        ratios = torch.exp(cur_lp - b_old_lp)
        surr1 = ratios * b_adv
        surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
        p_loss = -torch.min(surr1, surr2).mean()
        v_loss = 0.5 * ((val_p.squeeze() - torch.tensor(advantages + np.array(values[:-1]))) ** 2).mean()
        tot_loss = p_loss + 0.5 * v_loss
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()

    # 4. Now evaluate new policy vs old policy to check exact ratios & clipping status
    with torch.no_grad():
        new_logits, new_vals = net(b_states)
        new_dist = Categorical(logits=new_logits)
        new_lp = new_dist.log_prob(b_actions)
        new_ratios = torch.exp(new_lp - b_old_lp).numpy()

    # Select the requested number of evenly spaced or recent logs
    total_samples = len(states)
    selected_indices = np.linspace(max(0, total_samples - 60), total_samples - 1, num_logs, dtype=int)

    # Print Detailed Table Header
    print(f"{'Log#':<5} | {'Date':<10} | {'Ret_1d':<8} | {'RSI-14':<7} | {'MACD_h':<8} | {'ATR%':<6} | {'Action':<10} | {'V(s)':<7} | {'Adv (Â)':<8} | {'Ratio r(θ)':<10} | {'PPO Clipping Status'}")
    print("-" * 115)

    clipped_count = 0
    unclipped_count = 0

    for idx_num, i in enumerate(selected_indices, 1):
        dt = dates[i]
        
        # Raw features from feat_df
        row = feat_df.iloc[min(i, len(feat_df) - 1)]
        r1d = row.get("Ret_1d", 0.0) * 100.0
        rsi = row.get("RSI_14", 50.0)
        macd_h = row.get("MACD_hist", 0.0)
        close_p = row.get("Close", 1.0)
        atr_pct = (row.get("ATRr_14", 0.0) / close_p) * 100.0

        act_name = ACTION_NAMES[actions[i]]
        val_est = float(new_vals[i].item())
        adv_val = float(norm_advantages[i])
        ratio = float(new_ratios[i])

        # Clipping condition test
        is_clipped_high = ratio > (1.0 + clip_eps)
        is_clipped_low = ratio < (1.0 - clip_eps)
        is_clipped = is_clipped_high or is_clipped_low

        if is_clipped:
            clipped_count += 1
            if is_clipped_high:
                clip_str = f"CLIPPED [HIGH > 1.20] ({ratio:.3f})"
            else:
                clip_str = f"CLIPPED [LOW < 0.80]  ({ratio:.3f})"
        else:
            unclipped_count += 1
            clip_str = f"UNCLIPPED (Within Bound) [OK]"

        print(f"#{idx_num:<4} | {dt:<10} | {r1d:>+6.2f}% | {rsi:>7.1f} | {macd_h:>+8.2f} | {atr_pct:>5.2f}% | {act_name:<10} | {val_est:>+7.3f} | {adv_val:>+8.3f} | {ratio:>10.4f} | {clip_str}")

    print("-" * 115)
    print(f"\n  CLIPPING SUMMARY ACROSS {num_logs} LOGS:")
    print(f"    - Clipped Steps (Policy Shift Constrained) : {clipped_count} / {num_logs} ({clipped_count/num_logs*100:.1f}%)")
    print(f"    - Unclipped Steps (Standard Gradient Flow) : {unclipped_count} / {num_logs} ({unclipped_count/num_logs*100:.1f}%)")
    print(f"    - Trust Region Tolerance                   : 1.0 ± {clip_eps:.2f} (Interval: [0.800, 1.200])")
    print("=" * 115 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Feature & Clipping Inspector")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="NSE Stock Ticker (e.g. RELIANCE.NS, TCS.NS)")
    parser.add_argument("--logs", type=int, default=20, help="Number of steps/logs to inspect (default: 20)")
    parser.add_argument("--clip", type=float, default=0.20, help="PPO Surrogate epsilon (default: 0.20)")
    args = parser.parse_args()

    inspect_clipping(symbol=args.symbol, num_logs=args.logs, clip_eps=args.clip)
