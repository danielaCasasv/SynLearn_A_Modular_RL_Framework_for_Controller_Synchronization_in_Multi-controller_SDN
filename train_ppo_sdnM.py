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
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import MLPEncoderConfig, ActorCriticEncoderConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule

from gym_sdnM.envs.sdn_envM import SdnEnvM, max_tslots, n_domains

# --- Paths & Logging Setup ---
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
base_dir = os.path.join("runs_ppo_sdnM", timestamp)
chkpt_root = os.path.join(base_dir, "Checkpoints")
ray_results = os.path.join(base_dir, "Ray_Results")
os.makedirs(chkpt_root, exist_ok=True)
os.makedirs(ray_results, exist_ok=True)

logger = logging.getLogger("ppo_sdnM")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
fh = logging.FileHandler(os.path.join(base_dir, "log.txt"))
sh = logging.StreamHandler()
fh.setFormatter(formatter)
sh.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(sh)

logger.info(f"PyTorch: {torch.__version__}")
logger.info(f"Ray: {ray.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
logger.info(f"max_tslots={max_tslots}, n_domains={n_domains}")

# === Callback ===
class MetricsLogger(RLlibCallback):

    METRIC_KEYS = ["APC","mean_diff","mean_diff_norm","delta_diff","trend","sync_cost","urgency","misfire","delay"]
    def on_episode_created(self, *, episode, **kwargs):
        episode.custom_data["infos"] = {k: [] for k in self.METRIC_KEYS}
        episode.custom_data["rewards"] = []

    def on_episode_step(self, *, episode, metrics_logger, **kwargs):
        info = episode.get_infos()  # may be dict or list
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
        # Per-episode means
        for k, arr in episode.custom_data["infos"].items():
            if arr:
                metrics_logger.log_value(k+"_mean", float(np.mean(arr)), reduce="mean", clear_on_reduce=True)
        if episode.custom_data["rewards"]:
            metrics_logger.log_value("reward_sum", float(np.sum(episode.custom_data["rewards"])), reduce="mean", clear_on_reduce=True)

# --- Register env ---
def _make_env(cfg):
    # Optional: you can forward alpha/shaping if you want to sweep
    env = SdnEnvM(**cfg)
    return env
register_env("sdnM-v0", lambda cfg: TimeLimit(_make_env(cfg), max_episode_steps=max_tslots))

# --- Init Ray ---
ray.init(ignore_reinit_error=True, num_cpus=2, num_gpus=1 if torch.cuda.is_available() else 0)
logger.info("Ray initialized")

# --- RLModule spec ---
# Observation shape is 2*(n_domains-1)+2 (see SdnEnvM)
obs_dim = 2*(n_domains-1) + 2

rlm_spec = RLModuleSpec(
    module_class=DefaultPPOTorchRLModule,
    model_config={
        "actor_critic_encoder_config": ActorCriticEncoderConfig(
            base_encoder_config=MLPEncoderConfig(
                input_dims=[obs_dim],
                hidden_layer_dims=[256, 256],
                hidden_layer_activation="relu",
            )
        )
    },
    catalog_class=PPOCatalog,
)

# --- PPO config ---
config = (
    PPOConfig()
    .framework("torch")
    .resources(
        num_gpus=1 if torch.cuda.is_available() else 0,
        num_cpus_per_worker=1.0,
        num_gpus_per_worker=0.0
    )
    .environment(env="sdnM-v0", env_config={})
    .rl_module(rl_module_spec=rlm_spec)
    .callbacks(MetricsLogger)
    .training(
        train_batch_size=1024,     
        minibatch_size=256,
        num_epochs=10,
        clip_param=0.2,
        lambda_=0.95,              
        gamma=0.99,
        entropy_coeff=0.02,        
        vf_clip_param=100.0,       
    )
    .env_runners(
        num_env_runners=2,
        rollout_fragment_length="auto",
        batch_mode="complete_episodes",
        create_env_on_local_worker=True,
    )
    .reporting(keep_per_episode_custom_metrics=True)
)
algo = config.build()

num_iterations = 800
metrics_rows = []
INFO_METRICS = ["APC","mean_diff","mean_diff_norm","delta_diff","trend",
                "sync_cost","urgency","misfire","delay"]

for it in range(1, num_iterations + 1):
    result = algo.train()
    er = result.get("env_runners", {}) or {}

    row = {
        "iter": it,
        "num_episodes": er.get("num_episodes", np.nan),
        "num_episodes_lifetime": er.get("num_episodes_lifetime", np.nan),
        "episode_len_mean": er.get("episode_len_mean", np.nan),
        "num_env_steps_sampled": er.get("num_env_steps_sampled", np.nan),
        "num_env_steps_sampled_lifetime": er.get("num_env_steps_sampled_lifetime", np.nan),

        "episode_return_mean": er.get("episode_return_mean", np.nan),
        "episode_return_min": er.get("episode_return_min", np.nan),
        "episode_return_max": er.get("episode_return_max", np.nan),
        "reward_sum": er.get("reward_sum", np.nan),
    }
    for k in INFO_METRICS:
        row[f"{k}_mean"] = er.get(f"{k}_mean", np.nan)

    metrics_rows.append(row)

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

def pick_col(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None

ykey = pick_col(df, "episode_return_mean", "episode_reward_mean", "ret_mean")
if ykey is None:
    raise KeyError(f"No reward column found in df. Have: {list(df.columns)}")

plt.figure(); plt.plot(df["iter"], df[ykey]); plt.xlabel("Iteration"); plt.ylabel("Mean Return"); plt.title("PPO — Mean Return"); plt.savefig(os.path.join(ray_results, "mean_return.png")); plt.close()
# plt.figure(); plt.plot(df["iter"], df["episode_reward_mean"]); plt.xlabel("Iteration"); plt.ylabel("Mean Reward"); plt.title("PPO — Mean Reward"); plt.savefig(os.path.join(ray_results, "mean_reward.png")); plt.close()
plt.figure(); plt.plot(df["iter"], df["APC_mean"]); plt.xlabel("Iteration"); plt.ylabel("Avg APC"); plt.title("PPO — Avg APC"); plt.savefig(os.path.join(ray_results, "avg_apc.png")); plt.close()
plt.figure(); plt.plot(df["iter"], df["mean_diff_mean"]); plt.xlabel("Iteration"); plt.ylabel("Avg Diff"); plt.title("PPO — Avg Diff (Real vs Ctrl)"); plt.savefig(os.path.join(ray_results, "mean_diff.png")); plt.close()
plt.figure(); plt.plot(df["iter"], df["delta_diff_mean"]); plt.xlabel("Iteration"); plt.ylabel("Avg ΔDiff"); plt.title("PPO — Avg ΔDiff"); plt.savefig(os.path.join(ray_results, "avg_delta_diff.png")); plt.close()
plt.figure(); plt.plot(df["iter"], df["sync_cost_mean"]); plt.xlabel("Iteration"); plt.ylabel("Sync Cost"); plt.title("PPO — Sync Cost"); plt.savefig(os.path.join(ray_results, "sync_cost.png")); plt.close()

from pathlib import Path
from urllib.parse import urlparse

chkpt_root = Path("runs_ppo_sdnM") / timestamp / "Checkpoints"
chkpt_root.mkdir(parents=True, exist_ok=True)

ckpt_path = algo.save(checkpoint_dir=f"file://{chkpt_root.resolve()}")
logger.info(f"Checkpoint saved to {ckpt_path}")
ray.shutdown()
logger.info("Training complete")