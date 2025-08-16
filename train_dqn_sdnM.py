#!/usr/bin/env python3
import os
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import ray
from gymnasium.wrappers import TimeLimit
from ray.tune.registry import register_env
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.dqn import DQNConfig

from gym_sdnM.envs.sdn_envM import SdnEnvM, max_tslots, n_domains

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
base_dir = os.path.join("runs_dqn_sdnM", timestamp)
chkpt_root = os.path.join(base_dir, "Checkpoints")
ray_results = os.path.join(base_dir, "Ray_Results")
os.makedirs(chkpt_root, exist_ok=True)
os.makedirs(ray_results, exist_ok=True)

logger = logging.getLogger("dqn_sdnM")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(os.path.join(base_dir, "log.txt"))
sh = logging.StreamHandler()
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

class MetricsLogger(RLlibCallback):
    METRIC_KEYS = ["APC","mean_diff","mean_diff_norm","delta_diff","trend","sync_cost","urgency","misfire","delay"]
    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["infos"] = {k: [] for k in self.METRIC_KEYS}
        episode.custom_data["rewards"] = []

    def on_episode_step(self, *, episode, metrics_logger, **kwargs):
        info = episode.get_infos()  # lista o dict según stack
        if isinstance(info, dict):
            infos = [info]
        elif isinstance(info, (list, tuple)):
            infos = [i for i in info if isinstance(i, dict)]
        else:
            infos = []
        for inf in infos:
            for k in self.METRIC_KEYS:
                v = inf.get(k, None)
                if v is not None:
                    episode.custom_data["infos"][k].append(float(v))
        r = episode.get_rewards()
        if r is not None:
            if isinstance(r, (list, tuple, np.ndarray)):
                for ri in r:
                    episode.custom_data["rewards"].append(float(ri))
            else:
                episode.custom_data["rewards"].append(float(r))

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):
        for k, arr in episode.custom_data["infos"].items():
            if arr:
                metrics_logger.log_value(k+"_mean", float(np.mean(arr)), reduce="mean", clear_on_reduce=True)
        if episode.custom_data["rewards"]:
            metrics_logger.log_value("reward_sum", float(np.sum(episode.custom_data["rewards"])), reduce="mean", clear_on_reduce=True)

def _make_env(cfg):
    return SdnEnvM(**cfg)
register_env("sdnM-v0", lambda cfg: TimeLimit(_make_env(cfg), max_episode_steps=max_tslots))

ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=1 if torch.cuda.is_available() else 0)
logger.info("Ray initialized")

config = (
    DQNConfig()
    .framework("torch")
    .resources(num_gpus=1 if torch.cuda.is_available() else 0)
    .environment(env="sdnM-v0", env_config={})
    .callbacks(MetricsLogger)
    .training(
        lr=1e-3,
        train_batch_size=120,
        gamma=0.99,
        dueling=True,
        double_q=True,
        n_step=3
    )
    .env_runners(
        num_env_runners=1,
        rollout_fragment_length=30,
        batch_mode="complete_episodes",
        create_env_on_local_worker=True,
    )
    .reporting(keep_per_episode_custom_metrics=True)
)
algo = config.build()

num_iterations = 800
metrics_rows = []

# metrics that env logs each step 
INFO_METRICS = ["APC","mean_diff","mean_diff_norm","delta_diff","trend",
                "sync_cost","urgency","misfire","delay"]

for it in range(1, num_iterations + 1):
    result = algo.train()
    er = result.get("env_runners", {}) or {}

    # Collect iteration row – everything comes from env_runners
    row = {
        "iter": it,
        # episode/sample counters
        "num_episodes": er.get("num_episodes", np.nan),
        "num_episodes_lifetime": er.get("num_episodes_lifetime", np.nan),
        "episode_len_mean": er.get("episode_len_mean", np.nan),
        "num_env_steps_sampled": er.get("num_env_steps_sampled", np.nan),
        "num_env_steps_sampled_lifetime": er.get("num_env_steps_sampled_lifetime", np.nan),
        # returns
        "episode_return_mean": er.get("episode_return_mean", np.nan),
        "episode_return_min": er.get("episode_return_min", np.nan),
        "episode_return_max": er.get("episode_return_max", np.nan),
        "reward_sum": er.get("reward_sum", np.nan),
    }

    # Add every metric that was registered in env info 
    for k in INFO_METRICS:
        row[f"{k}_mean"] = er.get(f"{k}_mean", np.nan)

    metrics_rows.append(row)

    # Pretty log line
    logger.info(
        f"[it {it}] eps={row['num_episodes']} "
        f"ts_life={row['num_env_steps_sampled_lifetime']} "
        f"ret_mean={row['episode_return_mean']:.3f} "
        f"diff_mean={row.get('mean_diff_mean', np.nan)} "
        f"APC_mean={row.get('APC_mean', np.nan)}"
    )

df = pd.DataFrame(metrics_rows)

csv_path = os.path.join(ray_results, "metrics.csv")
df.to_csv(csv_path, index=False)
logger.info(f"Metrics saved to {csv_path}")

def _safe_plot(x, y_key, ylabel, title, fname):
    if y_key in df.columns and df[y_key].notna().any():
        plt.figure()
        plt.plot(df["iter"], df[y_key])
        plt.xlabel("Iteration"); plt.ylabel(ylabel); plt.title(title)
        plt.savefig(os.path.join(ray_results, fname)); plt.close()

_safe_plot("iter", "episode_return_mean", "Mean Return", "DQN — Mean Return", "mean_return.png")
_safe_plot("iter", "APC_mean", "Avg APC", "DQN — Avg APC", "avg_apc.png")
_safe_plot("iter", "mean_diff_mean", "Avg Diff", "DQN — Avg Diff", "mean_diff.png")
_safe_plot("iter", "delta_diff_mean", "Avg ΔDiff", "DQN — Avg ΔDiff", "avg_delta_diff.png")
_safe_plot("iter", "sync_cost_mean", "Sync Cost", "DQN — Sync Cost", "sync_cost.png")


chkpt_root = Path("runs_dqn_sdnM") / timestamp / "Checkpoints"
chkpt_root.mkdir(parents=True, exist_ok=True)

ckpt_path = algo.save(checkpoint_dir=f"file://{chkpt_root.resolve()}")
logger.info(f"Checkpoint saved to {ckpt_path}")
ray.shutdown()
logger.info("Training complete")



# import numpy as np
# from gym_sdnM.envs.sdn_envM import SdnEnvM, max_tslots
# env = SdnEnvM()
# obs, _ = env.reset()
# for t in range(max_tslots):
#     assert np.all(np.isfinite(obs)), f"NaN en obs t={t}"
#     obs, r, done, _, info = env.step(env.action_space.sample())
#     assert np.isfinite(r), f"NaN en reward t={t}"
#     if done: break
# print("OK: sin NaN")
