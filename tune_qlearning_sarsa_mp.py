import os
import time
import json
import argparse
import hashlib
import math
import random
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym

import gym_sdnM.envs.sdn_envM as sdn_envM
from QLearningAgent import QLearningAgent
from SarsaLambdaAgent import SarsaLambdaAgent

from train_qlearningM3 import trainRL, to_index
plt.rcParams['figure.figsize'] = (20, 8)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)

def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _parse_float_list(s: str):
    if not s:
        return None
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip() != ""]
    except Exception:
        return None


_NUM_KEY_PREC = 10  # decimales para floats en la normalización

def _round_if_float(x):
    # Coerce numpy scalars -> Python scalars
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, (np.integer,)):
        x = int(x)
    if isinstance(x, float):
        if math.isfinite(x):
            return float(f"{x:.{_NUM_KEY_PREC}f}")
        return x
    return x

def cfg_normalize(cfg: dict) -> dict:
    norm = {}
    for k, v in cfg.items():
        if isinstance(v, (list, tuple)):
            norm[k] = [_round_if_float(x) for x in v]
        else:
            norm[k] = _round_if_float(v)
    # Si el entorno expone estos parámetros, asegúrate de que existan en el cfg para hash estable:
    if hasattr(sdn_envM, "TAU_MISFIRE") and "tau_misfire" not in norm:
        norm["tau_misfire"] = _round_if_float(getattr(sdn_envM, "TAU_MISFIRE"))
    if hasattr(sdn_envM, "TAU_TREND") and "tau_trend" not in norm:
        norm["tau_trend"] = _round_if_float(getattr(sdn_envM, "TAU_TREND"))
    return norm

def cfg_stable_str(cfg: dict) -> str:
    return json.dumps(cfg_normalize(cfg), sort_keys=True, ensure_ascii=False)

def cfg_hash(cfg: dict, length: int = 10) -> str:
    h = hashlib.sha1(cfg_stable_str(cfg).encode("utf-8")).hexdigest()
    return h[:length]

def cfg_to_dirname(cfg: dict) -> str:
    agent = cfg.get("agent", "qlearning")
    base = f"{agent}_bins{cfg['bins']}_w{cfg['weights_change_time']}_s{cfg['sync_time']}"
    if agent == "sarsa" and "trace_decay" in cfg:
        base += f"_lam{_round_if_float(cfg['trace_decay'])}"
    return f"{base}_{cfg_hash(cfg)}"

def already_done(results_dir: str) -> bool:
    # Soporta legacy y nuevo nombre
    return any(os.path.exists(os.path.join(results_dir, fn))
               for fn in ("metrics.csv", "qlearning_metrics.csv"))


# Plots + Métricas
# -----------------------------
def save_pngs(results_dir: str, df: pd.DataFrame):
    plots = [
        ("mean_reward", "Mean Reward", "mean_reward.png"),
        ("mean_apc", "Avg APC", "avg_apc.png"),
        ("mean_diff", "Avg Diff (Real vs Ctrl)", "mean_diff.png"),
        ("mean_delta_diff", "Avg Delta Diff (Real vs Ctrl)", "mean_delta_diff.png"),
        ("mean_sync_cost", "Sync Cost", "sync_cost.png"),
    ]
    for col, title, fname in plots:
        if col in df.columns:
            plt.figure()
            plt.plot(df["episode"], df[col])
            plt.title(title)
            plt.xlabel("Episode")
            plt.ylabel(col)
            plt.savefig(os.path.join(results_dir, fname))
            plt.close()

def compute_objectives(df: pd.DataFrame) -> dict:
    metrics = {}
    n = len(df)
    last_vals = 100
    last_k = min(last_vals, n) if n > 0 else 0

    def last_mean(col):
        return float(np.mean(df[col].iloc[-last_k:])) if col in df.columns and last_k > 0 else np.nan

    metrics["obj_mean_reward_last"+str(last_vals)] = last_mean("mean_reward")
    metrics["mean_reward_all"] = float(np.mean(df["mean_reward"])) if "mean_reward" in df.columns else np.nan
    for col in ["mean_diff", "mean_delta_diff", "mean_sync_cost", "mean_diff_norm",
                "mean_apc", "mean_trend", "mean_urgency", "misfire_rate", "delay_rate"]:
        metrics[f"{col}_last{last_vals}"] = last_mean(col)

    action_cols = [c for c in df.columns if c.startswith("action_")]
    if action_cols:
        totals = df[action_cols].fillna(0).sum(axis=1)
        est_max_steps = int(totals.max()) if len(totals) else None
        metrics["sync_rate_est"] = float(np.mean(totals / est_max_steps)) if est_max_steps else np.nan
    else:
        metrics["sync_rate_est"] = np.nan

    return metrics

# Env e Agente
# -----------------------------
def set_env_constants_from_cfg(cfg: dict):
    sdn_envM.WEIGHTS_CHANGE_TIME = cfg["weights_change_time"]
    sdn_envM.SYNC_TIME = cfg["sync_time"]

    if hasattr(sdn_envM, "SYNC_COST"): sdn_envM.SYNC_COST = cfg.get("sync_cost", sdn_envM.SYNC_COST)
    if hasattr(sdn_envM, "ETA"):       sdn_envM.ETA       = cfg.get("eta", sdn_envM.ETA)
    if hasattr(sdn_envM, "BETA"):      sdn_envM.BETA      = cfg.get("beta", sdn_envM.BETA)
    if hasattr(sdn_envM, "GAMMA"):     sdn_envM.GAMMA     = cfg.get("gamma", sdn_envM.GAMMA)

    if hasattr(sdn_envM, "TAU_MISFIRE") and "tau_misfire" in cfg:
        sdn_envM.TAU_MISFIRE = cfg["tau_misfire"]
    if hasattr(sdn_envM, "TAU_TREND") and "tau_trend" in cfg:
        sdn_envM.TAU_TREND = cfg["tau_trend"]

def build_env(cfg: dict):
    env = gym.make(
        "gym_sdnM:sdnM-v0",
        alpha=cfg.get("alpha", 0.4),
        shaping_coeff=cfg.get("shaping_coeff", 0.05)
    ).unwrapped
    return env

def build_agent(cfg: dict, env, total_episodes: int):
    obs_dim = env.observation_space.shape[0]
    n_states = int(cfg["bins"]) ** int(obs_dim)
    n_actions = env.action_space.n

    if cfg.get("agent", "qlearning") == "sarsa":
        return SarsaLambdaAgent(
            trace_decay=float(cfg.get("trace_decay", 0.9)),
            gamma=float(cfg["gamma_agent"]),
            num_states=n_states,
            num_actions=n_actions,
            N0=float(cfg["N0"]),
            epsilon0=float(cfg["epsilon0"]),
            epsilon_min=float(cfg["epsilon_min"]),
            total_episodes=total_episodes
        )
    else:
        return QLearningAgent(
            gamma=float(cfg["gamma_agent"]),
            num_states=n_states,
            num_actions=n_actions,
            N0=float(cfg["N0"]),
            lr=float(cfg["lr"]),
            epsilon0=float(cfg["epsilon0"]),
            epsilon_min=float(cfg["epsilon_min"]),
            total_episodes=total_episodes
        )

# -----------------------------
# Training
# -----------------------------
def run_trial(cfg: dict,
              total_episodes: int,
              max_steps: int,
              seed: int,
              results_root: str,
              resume: bool) -> dict:
    
    np.random.seed(seed)
    random.seed(seed)

    trial_name = cfg_to_dirname(cfg)
    results_dir = os.path.join(results_root, trial_name)
    ensure_dir(results_dir)

    norm_cfg = cfg_normalize(cfg)
    write_json(os.path.join(results_dir, "params.json"), norm_cfg)

    trial_csv = os.path.join(results_dir, "metrics.csv")
    legacy_csv = os.path.join(results_dir, "qlearning_metrics.csv")
    if resume and (os.path.exists(trial_csv) or os.path.exists(legacy_csv)):
        try:
            df = pd.read_csv(trial_csv if os.path.exists(trial_csv) else legacy_csv)
            metrics = compute_objectives(df)
            metrics["elapsed_s"] = metrics.get("elapsed_s", None)
            write_json(os.path.join(results_dir, "metrics.json"), metrics)
            return {**norm_cfg, **metrics, "results_dir": results_dir, "status": "skipped", "resumed": True}
        except Exception:
            pass

    set_env_constants_from_cfg(cfg)
    env = build_env(cfg)
    agent = build_agent(cfg, env, total_episodes)

    start = time.time()
    outputs = trainRL(agent, env, total_episodes, max_steps, bins=int(cfg["bins"]), log_prefix="mp --")

    rewards = outputs[0]
    mean_apcs = outputs[1] if len(outputs) > 1 else []
    mean_diff = outputs[2] if len(outputs) > 2 else []
    mean_diff_norm = outputs[3] if len(outputs) > 3 else []
    mean_sync_cost = outputs[4] if len(outputs) > 4 else []
    action_counts = outputs[5] if len(outputs) > 5 else []
    mean_deltas = outputs[6] if len(outputs) > 6 else []
    mean_trend = outputs[7] if len(outputs) > 7 else []
    mean_urgency = outputs[8] if len(outputs) > 8 else []
    misfire_rate = outputs[9] if len(outputs) > 9 else []
    delay_rate = outputs[10] if len(outputs) > 10 else []

    # DataFrame episodios
    df_dict = {
        "episode": np.arange(1, len(rewards) + 1),
        "mean_reward": rewards
    }
    opt_cols = [
        ("mean_apc", mean_apcs),
        ("mean_diff", mean_diff),
        ("mean_diff_norm", mean_diff_norm),
        ("mean_sync_cost", mean_sync_cost),
        ("mean_delta_diff", mean_deltas),
        ("mean_trend", mean_trend),
        ("mean_urgency", mean_urgency),
        ("misfire_rate", misfire_rate),
        ("delay_rate", delay_rate),
    ]
    n = len(rewards)
    for name, vec in opt_cols:
        if isinstance(vec, (list, np.ndarray)) and len(vec) == n:
            df_dict[name] = vec
    df = pd.DataFrame(df_dict)

    if action_counts:
        action_df = pd.DataFrame(action_counts).fillna(0).astype(int)
        action_df.columns = [f"action_{c}" for c in action_df.columns]
        df = pd.concat([df, action_df.reset_index(drop=True)], axis=1)

    # Guardados
    df.to_csv(trial_csv, index=False)
    save_pngs(results_dir, df)

    metrics = compute_objectives(df)
    metrics["elapsed_s"] = round(time.time() - start, 2)
    write_json(os.path.join(results_dir, "metrics.json"), metrics)

    return {**norm_cfg, **metrics, "results_dir": results_dir, "status": "ok", "resumed": False}

# -----------------------------
# Grid: Para fazer busca de parâmetros
# -----------------------------
def default_param_grid() -> dict:
    # Mantener exactamente el grid pedido
    grid = {
        "bins": [3,4,5],
        "weights_change_time": [2,3],
        "sync_time": [2,3,4],
        "sync_cost": [0.01, 0.05, 0.1],
        "eta": [2.0],
        "beta": [0.6],
        "gamma": [0.1],
        "gamma_agent": [0.88],
        "lr": [0.2,0.05],
        "N0": [0.6],
        "epsilon0": [0.8],
        "epsilon_min": [0.005],
        "alpha": [0.4],
        "shaping_coeff": [0.05],
    }
    if hasattr(sdn_envM, "TAU_MISFIRE"):
        grid["tau_misfire"] = [0.005]
    if hasattr(sdn_envM, "TAU_TREND"):
        grid["tau_trend"] = [0.005]
    return grid

def iter_grid(param_grid: dict):
    keys = list(param_grid.keys())
    for values in product(*[param_grid[k] for k in keys]):
        yield dict(zip(keys, values))

# -----------------------------
# Multiproceso - Para rodar em paralelo
# -----------------------------
def _worker_entry(args_tuple):
    """
    Wrapper plano para que Pool.imap_unordered possa serializar a chamada
    """
    cfg, total_episodes, max_steps, seed, results_root, resume = args_tuple
    try:
        return run_trial(cfg, total_episodes, max_steps, seed, results_root, resume)
    except Exception as e:
        d = dict(cfg_normalize(cfg))
        d.update({"status": "fail", "error": str(e), "results_dir": None})
        return d

def run_pool(tasks,
             total_episodes: int,
             max_steps: int,
             seed: int,
             results_root: str,
             resume: bool,
             workers: int,
             chunksize: int,
             maxtasksperchild: int,
             outfile: str):

    ensure_dir(os.path.dirname(outfile) if os.path.dirname(outfile) else ".")
    rows = []
    i_seed = 0

    pool_args = []
    for cfg in tasks:
        pool_args.append((
            cfg, total_episodes, max_steps, seed + i_seed, results_root, resume
        ))
        i_seed += 1

    with Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
        for res in pool.imap_unordered(_worker_entry, pool_args, chunksize):
            rows.append(res)
            pd.DataFrame(rows).to_csv(outfile, index=False)

    df = pd.DataFrame(rows)
    best = None
    # Selección del best robusta
    score_col = "obj_mean_reward_last100"
    if score_col in df.columns and df[score_col].notna().any():
        ok = df[df[score_col].notna()]
        if len(ok):
            best = ok.sort_values(score_col, ascending=False).iloc[0].to_dict()
    if best is None and "mean_reward_all" in df.columns and df["mean_reward_all"].notna().any():
        ok = df[df["mean_reward_all"].notna()]
        if len(ok):
            best = ok.sort_values("mean_reward_all", ascending=False).iloc[0].to_dict()
    return df, best


def mark_best(results_root: str, best_row: dict, outfile: str, create_link: bool, agent: str = None):
    if not best_row:
        return
    best_info = {"best": best_row, "from": outfile, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_json(os.path.join(results_root, "BEST.json"), best_info)
    if create_link and best_row.get("results_dir"):
        try:
            link_name = f"best_trial_{agent}" if agent else "best_trial"
            link_path = os.path.join(results_root, link_name)
            if os.path.islink(link_path) or os.path.exists(link_path):
                try:
                    os.remove(link_path)
                except OSError:
                    pass
            os.symlink(best_row["results_dir"], link_path, target_is_directory=True)
        except Exception:
            pass

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Grid/Random search multiprocessing with resuming, stable hash and incremental saving."
    )
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per trial.")
    parser.add_argument("--steps", type=int, default=30, help="Steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--outfile", type=str, default="hparam_grid_mp.csv", help="Incremental summary CSV.")
    parser.add_argument("--results-root", type=str, default="experimentos_grid_mp",
                        help="Per-trial results folder.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                        help="Number of parallel processes.")
    parser.add_argument("--chunksize", type=int, default=4,
                        help="Chunk size for imap_unordered.")
    parser.add_argument("--maxtasksperchild", type=int, default=50,
                        help="Recycle child processes after N tasks (avoids memory leaks).")
    parser.add_argument("--no-resume", action="store_true",
                        help="If specified, do NOT skip already-done trials.")
    parser.add_argument("--limit", type=int, default=None,
                        help="(Optional) limit the number of combinations before sampling.")
    parser.add_argument("--all", action="store_true",
                        help="Force iterating over all pending combinations (default mode).")
    parser.add_argument("--random", type=int, default=None,
                        help="Random search without replacement: number of combinations to sample from pending ones.")
    parser.add_argument("--best-link", action="store_true",
                        help="Create 'best_trial' symlink to the best results_dir and save BEST.json.")
    parser.add_argument("--agent", choices=["qlearning", "sarsa"], default="qlearning",
                        help="Tabular algorithm to use.")
    parser.add_argument("--trace-decay", type=float, default=0.9,
                        help="Lambda for SARSA(λ). Ignored if --agent qlearning.")

    args = parser.parse_args()

    # Subfolder per agent and suffix in outfile
    agent = getattr(args, "agent", "qlearning")
    args.results_root = os.path.join(args.results_root, agent)
    base, ext = os.path.splitext(args.outfile)
    args.outfile = f"{base}_{agent}{ext or '.csv'}"

    resume = not args.no_resume
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1) Generate full grid
    grid = default_param_grid()
    combos = list(iter_grid(grid))

    # Inject the agent into each cfg (and lambda if SARSA)
    for c in combos:
        c["agent"] = agent
        if agent == "sarsa":
            c["trace_decay"] = args.trace_decay

    # 2) Filter by resume (skip done)
    if resume:
        orig = len(combos)
        combos = [cfg for cfg in combos
                  if not already_done(os.path.join(args.results_root, cfg_to_dirname(cfg)))]
        print(f"Resuming: {orig - len(combos)} combos were already done. Remaining {len(combos)}.")
    else:
        print(f"Resume disabled: cache will be ignored and results of existing combos will be overwritten.")

    # 3) Limit if requested (before random sampling)
    if args.limit is not None:
        combos = combos[:args.limit]
        print(f"Limit: running only {len(combos)} combinations (before sampling).")

    # 4) Mode selection: --random N (without replacement) or --all
    if args.random is not None:
        k = min(args.random, len(combos))
        combos = random.sample(combos, k)
        print(f"Random search without replacement: {k} combinations selected.")
    else:
        # --all is the default; keep combos as they are
        print(f"ALL mode: {len(combos)} pending combinations.")

    if not combos:
        print("No pending combinations. Nothing to do. ✨")
        # Even so, if a previous outfile exists, we report it
        if os.path.exists(args.outfile):
            df = pd.read_csv(args.outfile)
            print(f"Existing CSV: {args.outfile} (rows={len(df)})")
        return

    # 5) Launch pool
    t0 = time.time()
    df, best = run_pool(
        tasks=combos,
        total_episodes=args.episodes,
        max_steps=args.steps,
        seed=args.seed,
        results_root=args.results_root,
        resume=resume,
        workers=args.workers,
        chunksize=args.chunksize,
        maxtasksperchild=args.maxtasksperchild,
        outfile=args.outfile
    )
    print(f"Total time: {round(time.time()-t0, 2)} s")
    print("=== Best trial (by last 100 mean_reward) ===")
    print(json.dumps(best, indent=2, ensure_ascii=False) if best else "There were no valid trials.")
    print(f"Summary saved to: {args.outfile}")
    print(f"Total rows: {len(df)}")

    # 6) Mark best trial (optional)
    if best:
        mark_best(args.results_root, best, args.outfile, create_link=args.best_link, agent=agent)

if __name__ == "__main__":
    main()

# For training Sarsa instead of Qlearning, indicate --agent sarsa and --trace-decay 

# python tune_qlearning_mp.py --agent sarsa  --trace-decay 0.8   --episodes 800 --steps 30   
# --workers 6 --chunksize 1 --maxtasksperchild 30   --outfile grid_mp.csv --results-root exper_grid_mp --limit 200

