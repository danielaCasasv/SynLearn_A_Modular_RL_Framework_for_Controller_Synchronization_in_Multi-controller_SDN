import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import gymnasium as gym
from tqdm import tqdm
import argparse

from SarsaLambdaAgent import SarsaLambdaAgent
from QLearningAgent import QLearningAgent
import gym_sdnM.envs.sdn_envM as sdn_envM

plt.rcParams['figure.figsize'] = (14, 8)

def build_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["qlearning", "sarsa"], default="qlearning")
    p.add_argument("--trace-decay", type=float, default=0.9,
                   help="Lambda de SARSA(λ). Ignorado si --agent qlearning.")
    p.add_argument("--episodes", type=int, default=800)
    p.add_argument("--steps", type=int, default=30)
    return p.parse_args()

def generate_experiment_name(bins, weights_change, sync_time):
    return f"bins{bins}_weights{weights_change}_sync{sync_time}"

def to_index(state, bins):
    if isinstance(state, (tuple, list)) and len(state) > 0:
        state = state[0]
    if isinstance(state, dict):
        state = state.get('desync', next(iter(state.values())))
    arr = np.asarray(state).ravel()
    arr = np.floor(arr * bins).astype(int)
    arr = np.clip(arr, 0, bins - 1)
    idx = 0
    for v in arr:
        idx = idx * bins + int(v)
    return idx

def trainRL(agent, env, total_episodes, max_steps, bins, log_prefix=""):
    Ns = defaultdict(int)
    Nas = defaultdict(lambda: np.zeros(agent.num_actions))
    rewards, mean_apcs, mean_diff = [], [], []
    mean_diff_norm, mean_improv, mean_sync_cost, mean_deltas = [], [], [], []
    mean_trend, mean_urgency = [], []
    misfire_rate, delay_rate = [], []
    action_counts = []

    for e in tqdm(range(total_episodes), desc=f"{log_prefix} Training:"):
        Ns.clear(); Nas.clear()
        obs, _ = env.reset()
        s = to_index(obs, bins)
        Ns[s] += 1
        a = agent.choose_action_edecay(s, Ns, train=True, episode=e)
        Nas[s][a] += 1

        ep_actions = Counter()
        ep_reward = 0.0
        ep_apc, ep_diff, ep_diff_norm = [], [], []
        ep_sync_cost, ep_deltas = [], []
        ep_trend, ep_urgency, ep_misfire, ep_delay = [], [], [], []

        for t in range(max_steps):
            obs2, r, done, _, info = env.step(a)

            ep_apc.append(info.get('APC', 0.0))
            ep_diff.append(info.get('mean_diff', 0.0))
            ep_diff_norm.append(info.get('mean_diff_norm', 0.0))
            ep_sync_cost.append(info.get('sync_cost', 0.0))
            ep_trend.append(info.get('trend', 0.0))
            ep_urgency.append(info.get('urgency', 0.0))
            ep_misfire.append(info.get('misfire', 0))
            ep_delay.append(info.get('delay', 0))
            ep_deltas.append(info.get('delta_diff', 0.0))

            s2 = to_index(obs2, bins)
            Ns[s2] += 1
            a2 = agent.choose_action_edecay(s2, Ns, train=True, episode=e)
            Nas[s2][a2] += 1

            agent.update(s, s2, r, a, a2, Nas)
            ep_reward += r
            ep_actions[a] += 1

            s, a = s2, a2
            if done:
                break

        rewards.append(ep_reward)
        mean_apcs.append(np.mean(ep_apc) if ep_apc else 0.0)
        mean_diff.append(np.mean(ep_diff) if ep_diff else 0.0)
        mean_diff_norm.append(np.mean(ep_diff_norm) if ep_diff_norm else 0.0)
        mean_sync_cost.append(np.mean(ep_sync_cost) if ep_sync_cost else 0.0)
        mean_trend.append(np.mean(ep_trend) if ep_trend else 0.0)
        mean_urgency.append(np.mean(ep_urgency) if ep_urgency else 0.0)
        misfire_rate.append(np.mean(ep_misfire) if ep_misfire else 0.0)
        delay_rate.append(np.mean(ep_delay) if ep_delay else 0.0)
        action_counts.append(ep_actions)
        mean_deltas.append(np.mean(ep_deltas) if ep_deltas else 0.0)

    return rewards, mean_apcs, mean_diff, mean_diff_norm, mean_sync_cost, action_counts, mean_deltas, mean_trend, mean_urgency, misfire_rate, delay_rate

if __name__ == '__main__':
    args = build_cli()

    total_episodes = args.episodes
    max_steps = args.steps
    gamma = 0.9
    N0 = 0.6
    lr = 0.5

    bins_list = [4]
    weights_list = [2]
    sync_list = [4]
    all_combinations = [(b, w, s) for b in bins_list for w in weights_list for s in sync_list]

    summary = []
    for BINS, weights_change, sync_time in sorted(all_combinations):
        start_time = time.time()

        sdn_envM.WEIGHTS_CHANGE_TIME = weights_change
        sdn_envM.SYNC_TIME = sync_time

        # pode ajustar sync_cost e envelope_decay aqui:
        env = gym.make("gym_sdnM:sdnM-v0", alpha=0.4, shaping_coeff=0.05).unwrapped

        obs_dim = env.observation_space.shape[0]
        n_states = BINS ** obs_dim
        n_actions = env.action_space.n

        if args.agent == "sarsa":
            agent = SarsaLambdaAgent(
                trace_decay=args.trace_decay,
                gamma=gamma,
                num_states=n_states,
                num_actions=n_actions,
                N0=N0,
                epsilon0=0.8,
                epsilon_min=0.01,
                total_episodes=total_episodes
            )
        else:
            agent = QLearningAgent(
                gamma=gamma,
                num_states=n_states,
                num_actions=n_actions,
                N0=N0,
                lr=lr,
                epsilon0=0.8,
                epsilon_min=0.01,
                total_episodes=total_episodes
            )

        name = generate_experiment_name(BINS, weights_change, sync_time)
        agent_name = args.agent
        results_dir = os.path.join("experimentos_otro_state_2", agent_name, name)
        os.makedirs(results_dir, exist_ok=True)

        (rewards, mean_apcs, mean_diff, mean_diff_norm,
         mean_sync_cost, action_counts, mean_deltas,
         mean_trend, mean_urgency, misfire_rate, delay_rate) = trainRL(
            agent, env, total_episodes, max_steps, bins=BINS, log_prefix=name + " --"
        )

        df = pd.DataFrame({
            'episode': np.arange(1, total_episodes + 1),
            'mean_reward': rewards,
            'mean_apc': mean_apcs,
            'mean_diff': mean_diff,
            'mean_diff_norm': mean_diff_norm,
            'mean_sync_cost': mean_sync_cost,
            'mean_delta_diff': mean_deltas,
            'mean_trend': mean_trend,
            'mean_urgency': mean_urgency,
            'misfire_rate': misfire_rate,
            'delay_rate': delay_rate
        })

        action_df = pd.DataFrame(action_counts).fillna(0).astype(int)
        action_df.columns = [f"action_{col}" for col in action_df.columns]
        df = pd.concat([df, action_df], axis=1)
        df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

        # Plots
        plt.figure(); plt.plot(df['episode'], df['mean_reward']); plt.title('Mean Reward'); plt.savefig(os.path.join(results_dir, 'mean_reward.png')); plt.close()
        plt.figure(); plt.plot(df['episode'], df['mean_apc']); plt.title('Avg APC'); plt.savefig(os.path.join(results_dir, 'avg_apc.png')); plt.close()
        plt.figure(); plt.plot(df['episode'], df['mean_diff']);  plt.title('Avg Diff (Real vs Ctrl)'); plt.savefig(os.path.join(results_dir, 'mean_diff.png')); plt.close()
        plt.figure(); plt.plot(df['episode'], df['mean_delta_diff']);  plt.title('Avg Delta Diff (Real vs Ctrl)'); plt.savefig(os.path.join(results_dir, 'mean_delta_diff.png')); plt.close()
        plt.figure(); plt.plot(df['episode'], df['mean_sync_cost']); plt.title('Sync Cost'); plt.savefig(os.path.join(results_dir, 'sync_cost.png')); plt.close()

        elapsed = time.time() - start_time
        print(f"{name}|✔| em {elapsed:.2f}s")

        summary.append({
            "experiment": name,
            "bins": BINS,
            "weights": weights_change,
            "sync": sync_time,
            "mean_final_reward": float(np.mean(df['mean_reward'][-50:])),
            "mean_final_apc": float(np.mean(df['mean_apc'][-50:])),
            "mean_final_diff": float(np.mean(df['mean_diff'][-50:])),
            "mean_final_delta_diff": float(np.mean(df['mean_delta_diff'][-50:])),
            "action_0%": (df['action_0'].mean() / max_steps * 100) if 'action_0' in df else 0.0,
            "time_sec": round(elapsed, 1)
        })

    pd.DataFrame(summary).to_csv(f"experimentos_summary_{args.agent}.csv", index=False)
