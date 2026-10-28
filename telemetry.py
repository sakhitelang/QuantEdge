#!/usr/bin/env python3
"""
QuantEdge AI — Actor-Critic Reinforcement Learning Telemetry Runner
Outputs step-by-step policy probability shifts, layer-by-layer gradient norms,
and final converged loss metrics directly to the terminal for basic Actor-Critic.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

# Ensure project root is in path
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


def run_telemetry(symbol: str = "RELIANCE.NS", epochs: int = 5, cycles: int = 2):
    clean_sym = normalize_ticker(symbol)
    print("=" * 80)
    print(f"  QUANT EDGE ACTOR-CRITIC TRAINING TELEMETRY & POLICY DYNAMICS: {clean_sym}")
    print("=" * 80)

    # 1. Load data & compute features
    raw_df = load_ticker_data(clean_sym)
    feat_df = compute_features(raw_df)
    vix_val, nifty_ret = get_macro_context()

    mat = prepare_market_state_matrix(feat_df, vix_val=vix_val, nifty_ret=nifty_ret)
    asset_rets = feat_df["Ret_1d"].values.astype(np.float32)

    # 2. Initialize fresh system
    rl_sys = RLForecastingSystem()
    net = rl_sys.network

    # 3. Initial policy on latest market state
    latest_st = torch.tensor(np.append(mat[-1], [0.0, 0.0]), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        init_logits, init_val = net(latest_st)
        init_probs = torch.softmax(init_logits, dim=-1).squeeze().numpy()

    print(f"\n[1] INITIAL POLICY (Pre-Training Baseline):")
    print(f"    Critic Expected Baseline V(s): {init_val.item():+.4f}")
    for idx, name in ACTION_NAMES.items():
        bar = "#" * int(init_probs[idx] * 40)
        print(f"    - {name:<12}: {init_probs[idx]*100:>5.1f}%  | {bar}")

    # 4. Rollouts & Optimization Tracking
    env = IndianTradingEnv(mat, asset_rets)
    optimizer = rl_sys.optimizer

    print("\n" + "-" * 80)
    print(f"  CYCLE-BY-CYCLE ROLLOUT OPTIMIZATION (Basic Actor-Critic: gamma=0.99)")
    print("-" * 80)
    print(f" {'Cycle':<7} | {'Epoch':<6} | {'Total Loss':<11} | {'Policy Loss':<12} | {'Value Loss':<11} | {'Grad Norm':<10}")
    print("-" * 80)

    history = []
    for cycle in range(1, cycles + 1):
        state = env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        done = False
        while not done:
            st_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                act, _, val, _ = net.get_action_and_value(st_t, deterministic=False)
            nxt, rew, done, _ = env.step(act)
            states.append(state)
            actions.append(act)
            rewards.append(rew)
            next_states.append(nxt)
            dones.append(1.0 if done else 0.0)
            state = nxt

        b_states = torch.tensor(np.array(states), dtype=torch.float32)
        b_actions = torch.tensor(np.array(actions), dtype=torch.long)
        b_rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        b_next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        b_dones = torch.tensor(np.array(dones), dtype=torch.float32)

        dataset_size = len(states)
        batch_size = 64

        for epoch in range(1, epochs + 1):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            ep_t_loss, ep_p_loss, ep_v_loss, ep_gnorm = [], [], [], []

            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                mb = indices[start:end]

                logits, val_pred = net(b_states[mb])
                values = val_pred.squeeze(-1)

                with torch.no_grad():
                    _, next_val_pred = net(b_next_states[mb])
                    next_values = next_val_pred.squeeze(-1)
                    target_values = b_rewards[mb] + 0.99 * next_values * (1.0 - b_dones[mb])

                td_error = (target_values - values).detach()

                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(b_actions[mb])
                policy_loss = -(log_probs * td_error).mean()
                val_loss = 0.5 * ((values - target_values) ** 2).mean()

                loss = policy_loss + (0.5 * val_loss)

                optimizer.optimizer.zero_grad()
                loss.backward()

                # Grad norm
                raw_grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0).item())
                optimizer.optimizer.step()

                ep_t_loss.append(loss.item())
                ep_p_loss.append(policy_loss.item())
                ep_v_loss.append(val_loss.item())
                ep_gnorm.append(raw_grad_norm)

            avg_t = np.mean(ep_t_loss)
            avg_p = np.mean(ep_p_loss)
            avg_v = np.mean(ep_v_loss)
            avg_g = np.mean(ep_gnorm)
            history.append((cycle, epoch, avg_t, avg_p, avg_v, avg_g))
            print(f" Cycle {cycle:<2} | Ep {epoch:<3} | {avg_t:>11.4f} | {avg_p:>12.4f} | {avg_v:>11.4f} | {avg_g:>10.4f}")

    # 5. Final policy on latest market state
    with torch.no_grad():
        final_logits, final_val = net(latest_st)
        final_probs = torch.softmax(final_logits, dim=-1).squeeze().numpy()

    print("\n" + "=" * 80)
    print(f"[2] FINAL TRAINED POLICY & DELTA SHIFT:")
    print("=" * 80)
    print(f"  Critic Expected Value V(s): {final_val.item():+.4f} (Shift: {final_val.item() - init_val.item():+.4f})")
    print(f"\n  {'ACTION':<12} | {'INITIAL PROB':<14} | {'FINAL PROB':<12} | {'DELTA SHIFT':<14} | {'POLICY VECTOR'}")
    print("  " + "-" * 76)
    for idx, name in ACTION_NAMES.items():
        delta = final_probs[idx] - init_probs[idx]
        sign = "+" if delta >= 0 else ""
        bar = "#" * int(final_probs[idx] * 40)
        print(f"  {name:<12} | {init_probs[idx]*100:>12.1f}% | {final_probs[idx]*100:>10.1f}% | {sign}{delta*100:>11.1f}% | {bar}")

    # 6. Gradient diagnostics across layers
    print("\n" + "=" * 80)
    print("  LAYER-BY-LAYER GRADIENT NORMS & WEIGHT STATISTICS:")
    print("=" * 80)
    for name, param in net.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.data.norm(2).item()
            w_norm = param.data.norm(2).item()
            print(f"  - {name:<25}: Grad Norm L2 = {g_norm:>8.4f} | Weight Norm L2 = {w_norm:>8.4f}")

    print("\n" + "=" * 80)
    print(f"  FINAL CONVERGED LOSS METRICS:")
    print(f"    - Final Total Loss        : {history[-1][2]:.4f}")
    print(f"    - Final Policy Loss       : {history[-1][3]:.4f}")
    print(f"    - Final Value Loss (MSE)  : {history[-1][4]:.4f}")
    print(f"    - Mean Gradient Norm      : {history[-1][5]:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantEdge Actor-Critic Telemetry Tracker")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="NSE Ticker (e.g. RELIANCE.NS, TCS.NS)")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per rollout cycle")
    parser.add_argument("--cycles", type=int, default=2, help="Number of rollout cycles")
    args = parser.parse_args()

    run_telemetry(symbol=args.symbol, epochs=args.epochs, cycles=args.cycles)
