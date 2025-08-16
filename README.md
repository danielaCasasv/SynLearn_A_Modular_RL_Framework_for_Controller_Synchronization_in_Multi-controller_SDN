# SyncLearn — A Modular RL Framework for Controller Synchronization in Multi-Domain SDN

SyncLearn is a research codebase for studying **when and where to synchronize SDN controllers** under explicit budgets. It models non-stationary link weights, staleness signals, and synchronization cost, and casts the timing/selection problem as an executable MDP. The repo bundles **tabular RL**, **deep RL (DQN/PPO)**, and **deployable baselines** behind a unified Gymnasium/RLlib evaluation loop.&#x20;

---

## What is SyncLearn? 

* **Goal.** Keep a timely, coherent multi-controller view while respecting a sync budget, so end-to-end QoS is preserved. The framework exposes a discrete action space (noop or sync with a specific domain) and returns per-decision metrics: reward, path-cost discrepancy (and step-wise improvement), and synchronization cost.&#x20;
* **MDP.** State encodes per-domain staleness, per-domain cost mismatch, an EMA trend of discrepancy, and time since last sync; the reward trades improvement against sync cost with safeguards against unproductive syncs and harmful delays.&#x20;
* **Findings (paper).** On a 5-domain topology, **DQN** achieved the best cost–benefit: ≈38% lower average discrepancy than the strongest baseline (Periodic-k) while keeping moderate synchronization and the highest reward.&#x20;

---

## Repository structure

* `Agent.py` – Base class for epsilon-greedy agents with both **count-based** and **episode-based** epsilon decay utilities. Holds the tabular Q-table and action selection helpers.&#x20;
* `QLearningAgent.py` – Tabular **Q-Learning** with an adaptive learning rate `α=1/N(s,a)` (falls back to a fixed LR initially). Inherits exploration from `Agent`.&#x20;
* `SarsaLambdaAgent.py` – Tabular **SARSA(λ)** with *accumulating eligibility traces* and `α=1/N(s,a)`; uses `Agent`’s epsilon-decay strategies.&#x20;
* `train_qlearningM3.py` – Minimal training loop for Q-Learning/SARSA over the **`gym_sdnM:sdnM-v0`** environment. Handles discretization (`to_index`), logging of APC/Diff/ΔDiff/Sync-cost, and PNG plots. CLI: `--agent {qlearning,sarsa} --trace-decay --episodes --steps`.&#x20;
* `tune_qlearning_sarsa_mp.py` – **Multiprocess grid/random search** over tabular hyperparameters with resume, stable hashing of configs, CSV summaries, and symlink to best trial. Saves per-trial `metrics.csv` and plots.&#x20;
* `train_dqn_sdnM.py` – **RLlib DQN** (dueling, double, n-step) training harness with custom callback logging env metrics (APC, Diff, ΔDiff, Sync-cost, etc.), rolling plots, and checkpointing.&#x20;
* `train_ppo_sdnM.py` – **RLlib PPO** training with a Torch MLP actor-critic encoder, same logging/plots/checkpoints as DQN.&#x20;
* `baselines.py` – Deployable baselines & CLI: **Random**, **Periodic-k (round-robin sync)**, **Greedy-ϕ** (largest per-domain mismatch, tie-break by staleness), and an **Oracle one-step lookahead** (diagnostic). Writes CSV + plots.&#x20;
* `plots.py` – Convenience scripts to parse logs/metrics and generate comparison charts across algorithms (DQN, PPO, Q-Learning, SARSA).&#x20;

> **Note.** The Gymnasium environment **`gym_sdnM.envs.sdn_envM`** (with `SdnEnvM`, `max_tslots`, `n_domains`) is an external module this repo depends on. Ensure it is installed/available on `PYTHONPATH`. The training scripts assume its observation layout and metrics.

---

## Getting started

### Requirements

* Python 3.9+
* `gymnasium`, `numpy`, `pandas`, `matplotlib`
* **RLlib/Ray** (for DQN/PPO): `ray[rllib]` + PyTorch
* **Custom env** `gym_sdnM` providing `sdnM-v0` (install your local package or add the source path)

### Installation

```bash
# install the essentials
pip install gymnasium numpy pandas matplotlib torch "ray[rllib]"
# make sure gym_sdnM is importable, e.g.:
pip install -e /path/to/gym_sdnM
```

---

## Quickstarts

### Tabular agents (Q-Learning / SARSA(λ))

```bash
# Train Q-Learning
python train_qlearningM3.py --agent qlearning --episodes 800 --steps 30

# Train SARSA(λ)
python train_qlearningM3.py --agent sarsa --trace-decay 0.9 --episodes 800 --steps 30
```

**Hyper-parameter search (multiprocess)**

```bash
# Exhaustive (or use --random N for sampling)
python tune_qlearning_sarsa_mp.py --agent qlearning --episodes 200 --steps 30 --workers 6 --chunksize 1 --maxtasksperchild 30 --outfile grid_mp.csv --results-root exper_grid_mp --limit 200

# SARSA sweep example
python tune_qlearning_sarsa_mp.py --agent sarsa  --trace-decay 0.8 --episodes 800 --steps 30 --workers 6 --chunksize 1 --maxtasksperchild 30 --outfile grid_mp.csv --results-root exper_grid_mp --limit 200
```

Saves per-trial folders under `exper_grid_mp/<agent>/...`, plus an incremental summary CSV and optional `best_trial` symlink.&#x20;

### Deep RL (DQN / PPO via RLlib)

```bash
# DQN
python train_dqn_sdnM.py

# PPO
python train_ppo_sdnM.py
```

Both register `sdnM-v0`, log per-episode means for **APC**, **Diff**, **ΔDiff**, **Sync cost**, **misfire/delay**, and produce plots + checkpoints in timestamped `runs_*_sdnM/` directories.

### Baselines

```bash
# Random / Periodic-k / Greedy-phi / Oracle
python baselines.py --baseline random   --episodes 300 --steps 30
python baselines.py --baseline periodic --k 3       --episodes 300 --steps 30
python baselines.py --baseline greedy   --phi-th 0.02 --episodes 300 --steps 30
python baselines.py --baseline oracle   --episodes 100 --steps 30
```
