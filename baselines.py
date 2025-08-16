#!/usr/bin/env python3
"""
Baselines for SyncLearn in gym_sdnM.envs.sdn_envM.

Implements the baselines described in the paper draft:
  • Random: pick noop or any controller uniformly at each decision.
  • Periodic-k: every k decisions do a round-robin sync across controllers; otherwise noop.
  • Greedy-phi (thresholded): pick controller with largest current mismatch ϕ̂_{i,t}
    (we use the per-domain normalized diff from the observation); break ties by staleness τ̂_{i,t}
    (we use the per-domain desync from the observation). If all ϕ̂ below a threshold, do noop.
  • Oracle (optional, best-effort): one-step lookahead via a deep-copied env; choose action
    that yields lowest APC after a single step. This is informative only, may be slow.

Outputs a CSV compatible with your tabular scripts (columns like mean_reward, mean_apc,
mean_diff, mean_delta_diff, mean_sync_cost, etc.) plus simple PNG plots.

Usage examples:
  python baseline_eval_sdnM.py --baseline random --episodes 300 --steps 30
  python baseline_eval_sdnM.py --baseline periodic --k 3 --episodes 300 --steps 30
  python baseline_eval_sdnM.py --baseline greedy --phi-th 0.02 --episodes 300 --steps 30
  python baseline_eval_sdnM.py --baseline oracle --episodes 100 --steps 30

Notes:
  • Uses env observation layout from SdnEnvM._get_observation():
      [desync_{B..E}, diffs_norm_{B..E}, trend_norm, since_last_sync]
    where len(desync_*) = len(diffs_norm_*) = n_domains-1.
  • Action mapping uses ACTION_MAP with 0 = noop, 1..=sync B..E.
"""
from __future__ import annotations
import os
import time
import json
import argparse
import random
import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym

# Environment + constants
import gym_sdnM.envs.sdn_envM as sdn_envM
from gym_sdnM.envs.sdn_envM import max_tslots, n_domains

plt.rcParams["figure.figsize"] = (16, 6)

# ----------------------
# Filesystem helpers
# ----------------------
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def save_pngs(results_dir: str, df: pd.DataFrame):
    plots = [
        ("mean_reward", "Mean Reward", "mean_reward.png"),
        ("mean_apc", "Avg APC", "avg_apc.png"),
        ("mean_diff", "Avg Diff (Real vs Ctrl)", "mean_diff.png"),
        ("mean_delta_diff", "Avg ΔDiff", "mean_delta_diff.png"),
        ("mean_sync_cost", "Sync Cost", "sync_cost.png"),
    ]
    for col, title, fname in plots:
        if col in df.columns and df[col].notna().any():
            plt.figure()
            plt.plot(df["episode"], df[col])
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel(col)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, fname))
            plt.close()


# ----------------------
# Policies (baselines)
# ----------------------
@dataclass
class PolicyState:
    rr_idx: int = 0  # for periodic round-robin


class BasePolicy:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.state = PolicyState()

    def reset(self):
        self.state = PolicyState()

    def act(self, obs: np.ndarray, t: int) -> int:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    def act(self, obs: np.ndarray, t: int) -> int:
        return int(np.random.randint(self.n_actions))


class PeriodicPolicy(BasePolicy):
    def __init__(self, n_actions: int, k: int):
        super().__init__(n_actions)
        assert n_actions >= 2, "Need at least noop + 1 controller"
        self.k = max(1, int(k))

    def act(self, obs: np.ndarray, t: int) -> int:
        # Every k steps: sync using round-robin target; otherwise noop (0)
        if (t + 1) % self.k == 0:
            # Controllers map to actions 1..n_actions-1
            act = 1 + (self.state.rr_idx % (self.n_actions - 1))
            self.state.rr_idx += 1
            return act
        return 0  # noop


class GreedyPhiPolicy(BasePolicy):
    def __init__(self, n_actions: int, phi_threshold: float = 0.02):
        super().__init__(n_actions)
        self.phi_threshold = float(phi_threshold)

    def act(self, obs: np.ndarray, t: int) -> int:
        # Observation: [desync_(n-1), diffs_norm_(n-1), trend, since_last_sync]
        m = n_domains - 1
        desync = np.array(obs[:m], dtype=float)
        diffs = np.array(obs[m:2 * m], dtype=float)
        if diffs.size == 0:
            return 0
        best_phi = float(np.max(diffs))
        if best_phi < self.phi_threshold:
            return 0  # noop
        # Tie-break by desync; pick controller index (0..m-1) with largest (phi, desync)
        # We implement lexicographic by combining with tiny epsilon.
        eps = 1e-6
        j = int(np.argmax(diffs + eps * desync))
        return 1 + j  # action ids offset by 1


class OracleOneStepPolicy(BasePolicy):
    """One-step lookahead (best-effort). Very slow; informative only.

    Deep-copies the env, applies each candidate action, runs a step,
    and selects the one that gives the minimum APC (from info["APC"]).
    Falls back to Greedy-phi if deepcopy fails.
    """

    def __init__(self, n_actions: int, greedy_fallback: GreedyPhiPolicy):
        super().__init__(n_actions)
        self.greedy_fallback = greedy_fallback

    def act_from_env(self, env, obs: np.ndarray, t: int) -> int:
        try:
            best_act, best_apc = 0, float("inf")
            for a in range(self.n_actions):
                env_copy = copy.deepcopy(env)
                # Step once with candidate action
                _, _, done, _, info = env_copy.step(a)
                apc = float(info.get("APC", np.inf))
                if apc < best_apc:
                    best_apc, best_act = apc, a
            return int(best_act)
        except Exception:
            return self.greedy_fallback.act(obs, t)

    def act(self, obs: np.ndarray, t: int) -> int:
        raise RuntimeError("Use act_from_env(env, obs, t) with access to env instance.")


# ----------------------
# Rollout / Evaluation
# ----------------------
INFO_KEYS = [
    "APC", "mean_diff", "mean_diff_norm", "delta_diff", "trend",
    "sync_cost", "urgency", "misfire", "delay"
]


def run_episode(env, policy: BasePolicy, max_steps: int, use_oracle: bool = False) -> Tuple[dict, dict]:
    obs, _ = env.reset()
    policy.reset()

    rewards: List[float] = []
    info_acc: Dict[str, List[float]] = {k: [] for k in INFO_KEYS}
    action_counts = np.zeros(env.action_space.n, dtype=int)

    for t in range(max_steps):
        if use_oracle and isinstance(policy, OracleOneStepPolicy):
            a = policy.act_from_env(env, obs, t)
        else:
            a = policy.act(obs, t)
        action_counts[a] += 1

        obs, r, done, _, info = env.step(a)
        rewards.append(float(r))
        for k in INFO_KEYS:
            v = info.get(k, None)
            if v is not None:
                info_acc[k].append(float(v))
        if done:
            break

    # Per-episode means
    ep = {
        "mean_reward": float(np.mean(rewards)) if rewards else np.nan,
        "mean_apc": float(np.mean(info_acc["APC"])) if info_acc["APC"] else np.nan,
        "mean_diff": float(np.mean(info_acc["mean_diff"])) if info_acc["mean_diff"] else np.nan,
        "mean_diff_norm": float(np.mean(info_acc["mean_diff_norm"])) if info_acc["mean_diff_norm"] else np.nan,
        "mean_delta_diff": float(np.mean(info_acc["delta_diff"])) if info_acc["delta_diff"] else np.nan,
        "mean_trend": float(np.mean(info_acc["trend"])) if info_acc["trend"] else np.nan,
        "mean_sync_cost": float(np.mean(info_acc["sync_cost"])) if info_acc["sync_cost"] else np.nan,
        "mean_urgency": float(np.mean(info_acc["urgency"])) if info_acc["urgency"] else np.nan,
        "misfire_rate": float(np.mean(info_acc["misfire"])) if info_acc["misfire"] else np.nan,
        "delay_rate": float(np.mean(info_acc["delay"])) if info_acc["delay"] else np.nan,
    }
    actions = {f"action_{i}": int(c) for i, c in enumerate(action_counts.tolist())}
    return ep, actions


def set_env_constants_from_args(args):
    # Only override if provided (to stay consistent with your RL runs by default)
    if args.weights_change_time is not None:
        sdn_envM.WEIGHTS_CHANGE_TIME = int(args.weights_change_time)
    if args.sync_time is not None:
        sdn_envM.SYNC_TIME = int(args.sync_time)
    if args.sync_cost is not None:
        sdn_envM.SYNC_COST = float(args.sync_cost)
    if args.eta is not None:
        sdn_envM.ETA = float(args.eta)
    if args.beta is not None:
        sdn_envM.BETA = float(args.beta)
    if args.gamma is not None:
        sdn_envM.GAMMA = float(args.gamma)
    if args.tau_misfire is not None and hasattr(sdn_envM, "TAU_MISFIRE"):
        sdn_envM.TAU_MISFIRE = float(args.tau_misfire)
    if args.tau_trend is not None and hasattr(sdn_envM, "TAU_TREND"):
        sdn_envM.TAU_TREND = float(args.tau_trend)


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines on SdnEnvM.")
    parser.add_argument("--baseline", choices=["random", "periodic", "greedy", "oracle"], default="random")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--steps", type=int, default=max_tslots)
    parser.add_argument("--seed", type=int, default=0)

    # Baseline params
    parser.add_argument("--k", type=int, default=3, help="k for periodic-k baseline")
    parser.add_argument("--phi-th", type=float, default=0.02, help="ϕ threshold for greedy-phi")

    # Env knobs (optional; keep None to match RL runs defaults)
    parser.add_argument("--weights-change-time", type=int, default=None)
    parser.add_argument("--sync-time", type=int, default=None)
    parser.add_argument("--sync-cost", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau-misfire", type=float, default=None)
    parser.add_argument("--tau-trend", type=float, default=None)

    parser.add_argument("--outdir", type=str, default=None,
                        help="Where to save results. Default: runs_baselines_sdnM/<timestamp>/<baseline>")

    args = parser.parse_args()

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Env constants (optional overrides)
    set_env_constants_from_args(args)

    # Output directory
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    base = args.outdir or os.path.join("runs_baselines_sdnM", ts, args.baseline)
    ensure_dir(base)

    # Build env
    env = gym.make(
        "gym_sdnM:sdnM-v0",
        alpha=0.4,
        shaping_coeff=0.05,
    ).unwrapped

    n_actions = env.action_space.n

    # Policy
    if args.baseline == "random":
        policy = RandomPolicy(n_actions)
        oracle_flag = False
    elif args.baseline == "periodic":
        policy = PeriodicPolicy(n_actions, k=args.k)
        oracle_flag = False
    elif args.baseline == "greedy":
        policy = GreedyPhiPolicy(n_actions, phi_threshold=args.phi_th)
        oracle_flag = False
    else:  # oracle
        greedy = GreedyPhiPolicy(n_actions, phi_threshold=args.phi_th)
        policy = OracleOneStepPolicy(n_actions, greedy_fallback=greedy)
        oracle_flag = True

    # Run episodes
    rows = []
    for ep in range(1, args.episodes + 1):
        ep_metrics, action_counts = run_episode(env, policy, max_steps=args.steps, use_oracle=oracle_flag)
        row = {"episode": ep, **ep_metrics, **action_counts}
        rows.append(row)
        if ep % 10 == 0:
            print(f"[ep {ep}] reward={row['mean_reward']:.3f} diff={row.get('mean_diff', np.nan)} APC={row.get('mean_apc', np.nan)}")

    df = pd.DataFrame(rows)

    # Save artifacts
    csv_path = os.path.join(base, f"{args.baseline}_metrics.csv")
    df.to_csv(csv_path, index=False)
    with open(os.path.join(base, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    save_pngs(base, df)

    print(f"Saved CSV to: {csv_path}")
    print(f"Saved PNGs in: {base}")


if __name__ == "__main__":
    main()

# python baseline_eval_sdnM.py --baseline random --episodes 300 --steps 30
# python baseline_eval_sdnM.py --baseline periodic --k 3 --episodes 300 --steps 30
# python baseline_eval_sdnM.py --baseline greedy --phi-th 0.02 --episodes 300 --steps 30
# python baseline_eval_sdnM.py --baseline oracle --episodes 100 --steps 30
# --weights-change-time --sync-time --sync-cost --eta --beta --gamma --tau-misfire --tau-trend
