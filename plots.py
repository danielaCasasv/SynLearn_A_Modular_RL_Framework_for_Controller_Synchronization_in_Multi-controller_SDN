Extra: por qué PPO/DQN podrían ganar a Q/SARSA aquí
Tu reward premia mejoras de mean_diff y penaliza sync sin mejora (misfire) o demora (delay), con shaping por trend y urgency. Eso induce un trade‑off temporal con señales densas pero ruidosas. Algoritmos con política explícita (PPO) o con Q‑targets estabilizados (DQN dueling+double, n‑step) suelen navegar mejor estas señales que tabulares puramente exploratorios, sobre todo con estado continuo y “envelopes” que cambian. (Ver cómo construyes reward y los flags de misfire/delay.)



# Parse /mnt/data/log.txt and plot three separate charts:
# X-axis: it
# Y-axes: APC_mean, diff_mean, ret_mean (one chart per metric)

import re
import matplotlib.pyplot as plt

path = "/mnt/data/log.txt"

with open(path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

pattern = re.compile(
    r"\[it\s+(\d+)\].*?ret_mean=([\-0-9\.eE]+)\s+diff_mean=([\-0-9\.eE]+)\s+APC_mean=([\-0-9\.eE]+)"
)

rows = []
for m in pattern.finditer(text):
    it = int(m.group(1))
    ret_mean = float(m.group(2))
    diff_mean = float(m.group(3))
    apc_mean = float(m.group(4))
    rows.append((it, ret_mean, diff_mean, apc_mean))

# Sort by iteration
rows.sort(key=lambda x: x[0])

# If no rows found, raise a helpful message
if not rows:
    raise ValueError("No se encontraron líneas que coincidan con el patrón en log.txt.")

its = [r[0] for r in rows]
ret = [r[1] for r in rows]
diff = [r[2] for r in rows]
apc = [r[3] for r in rows]

# Create three separate figures (no subplots), do not specify colors.
plt.figure()
plt.plot(its, apc)
plt.title("it vs APC_mean")
plt.xlabel("it")
plt.ylabel("APC_mean")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(its, diff)
plt.title("it vs diff_mean")
plt.xlabel("it")
plt.ylabel("diff_mean")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(its, ret)
plt.title("it vs ret_mean")
plt.xlabel("it")
plt.ylabel("ret_mean")
plt.grid(True)
plt.tight_layout()
plt.show()

######################################################
# Re-run a robust analysis of the four metrics files and regenerate comparison charts.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
from caas_jupyter_tools import display_dataframe_to_user

base = Path("/mnt/data")

files = {
    "DQN": base / "metrics_dqn.csv",
    "PPO": base / "metrics_ppo.csv",
    "QLearning": base / "metrics_qlearning.csv",
    "SARSA": base / "metrics_sarsa.csv",
}

loaded: Dict[str, pd.DataFrame] = {}
errors = {}

for algo, path in files.items():
    try:
        if path.exists():
            df = pd.read_csv(path)
            df["__source_path"] = str(path)
            loaded[algo] = df
        else:
            errors[algo] = f"File not found: {path}"
    except Exception as e:
        errors[algo] = f"Failed to read {path}: {e}"

def standardize(df: pd.DataFrame, algo: str) -> pd.DataFrame:
    """Map differing column names into a shared schema."""
    out = df.copy()

    # step
    step_col = None
    for candidate in ["episode", "iter", "iteration", "step", "t", "epochs"]:
        if candidate in out.columns:
            step_col = candidate
            break
    if step_col is None:
        out["step"] = np.arange(1, len(out) + 1)
    else:
        out["step"] = out[step_col]

    # mean reward
    reward_col = None
    for candidate in ["mean_reward", "episode_reward_mean", "reward_mean", "return_mean"]:
        if candidate in out.columns:
            reward_col = candidate
            break
    out["mean_reward_std"] = np.nan
    if reward_col is not None:
        out["mean_reward"] = out[reward_col]
    else:
        # Some logs may only have per-iteration reward already summarized; if not present, keep NaN
        out["mean_reward"] = np.nan

    # Avg APC (application path cost proxy)
    apc_col = None
    for candidate in ["avg_apc", "APC_mean", "mean_apc", "apc_mean"]:
        if candidate in out.columns:
            apc_col = candidate
            break
    out["avg_apc"] = out[apc_col] if apc_col is not None else np.nan

    # Mean diff (real vs controller)
    mdiff_col = None
    for candidate in ["mean_diff", "mean_diff_mean"]:
        if candidate in out.columns:
            mdiff_col = candidate
            break
    out["mean_diff"] = out[mdiff_col] if mdiff_col is not None else np.nan

    # Keep only relevant columns
    keep = ["step", "mean_reward", "avg_apc", "mean_diff"]
    out = out[keep].copy()
    out["algo"] = algo
    return out

# Build combined frame
frames = []
for algo, df in loaded.items():
    try:
        frames.append(standardize(df, algo))
    except Exception as e:
        errors[algo] = f"Standardization failed: {e}"

if not frames:
    combined = pd.DataFrame(columns=["algo","step","mean_reward","avg_apc","mean_diff"])
else:
    combined = pd.concat(frames, ignore_index=True).sort_values(["algo","step"])

# Compute last-window summary (last 50 rows per algo, or all if less)
def last_window(df: pd.DataFrame, n=50) -> pd.DataFrame:
    rows = []
    for algo, g in df.groupby("algo", sort=False):
        tail = g.tail(min(n, len(g)))
        rows.append({
            "algo": algo,
            "steps_considered": len(tail),
            "final_mean_reward": float(tail["mean_reward"].mean(skipna=True)),
            "final_avg_apc": float(tail["avg_apc"].mean(skipna=True)),
            "final_mean_diff": float(tail["mean_diff"].mean(skipna=True)),
        })
    return pd.DataFrame(rows)

summary_df = last_window(combined, n=50)

# Display the summary to the user
display_dataframe_to_user("RL Summary (last 50 steps)", summary_df)

# Plot helper
def line_plot(df: pd.DataFrame, y_col: str, title: str, fname: str) -> Optional[str]:
    sub = df.dropna(subset=[y_col])
    if sub.empty:
        return None
    plt.figure()
    for algo, g in sub.groupby("algo"):
        g_sorted = g.sort_values("step")
        plt.plot(g_sorted["step"], g_sorted[y_col], label=algo)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(y_col.replace("_", " ").title())
    plt.legend()
    out_path = base / fname
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return str(out_path)

paths = {}
paths["mean_reward"] = line_plot(combined, "mean_reward", "Mean Reward over Time", "comparison_mean_reward.png")
paths["avg_apc"] = line_plot(combined, "avg_apc", "Avg APC over Time", "comparison_avg_apc.png")
paths["mean_diff"] = line_plot(combined, "mean_diff", "Avg Diff (Real vs Ctrl) over Time", "comparison_mean_diff.png")

# Return a short structured summary and any file paths for convenience
{
    "errors": errors,
    "generated_files": {k: v for k, v in paths.items() if v is not None},
    "combined_head": combined.head(3).to_dict(orient="records"),
    "summary": summary_df.to_dict(orient="records")
}
