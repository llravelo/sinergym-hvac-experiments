# 43008 Reinforcement Learning — Cadence (Sinergym HVAC)

DQN and PPO HVAC controllers for Sinergym's 5Zone building (EnergyPlus), compared against rule-based, fixed-setpoint and random baselines. Everything lives in one notebook, `notebooks/cadence_hvac_dqn_ppo_v2.ipynb`: environment, reward, training, evaluation and the GUI bundle export.

Python, EnergyPlus and Sinergym run inside Docker. The host only needs Docker (with the Compose plugin, `docker compose version`) and `make`.

## Run

```bash
git clone git@github.com:llravelo/sinergym-hvac-experiments.git
cd sinergym-hvac-experiments
make build   # first time only: pulls the Sinergym base image
make run     # executes the notebook headless, saves outputs into it
```

`make run` trains both agents (30 simulated years each), then evaluates every policy for one year in each climate, so it takes hours. Run it inside `tmux`/`screen` or with `nohup` so a dropped terminal doesn't kill it. Trained models and evaluation episodes are cached under `results/`. Re-running skips finished steps unless `RETRAIN`/`REEVALUATE` are set to `True` in the notebook.

## Outputs (`results/cadence_v2/`)

| Path | Contents |
|---|---|
| `dqn/`, `ppo/` | Final model, `VecNormalize` stats, per-year training log |
| `eval_episodes/` | Evaluation traces and decisions (Parquet) |
| `figures/` | Plots |
| `evaluation_metrics.csv`, `summary.csv` | Metrics tables |
| `gui_bundle/cadence_v2.zip` | Dashboard input: unzip into the GUI repo's `web/public/bundles/cadence_v2/` |

The bundle format is defined in `tools/cadence_bundle.py`, a copy of the GUI repo's file. Keep the two in sync.

## Other commands

| Command | What it does |
|---|---|
| `make lab` | JupyterLab at the printed `http://localhost:8888/...` URL |
| `make shell` | Bash shell in a throwaway container |
| `make clean` | Deletes leftover `Eplus-*-res*/` simulation folders (root-owned, so this uses the container) |

The container runs as root, so files it writes into the repo (`results/`, `Eplus-*` folders) are owned by root on the host.
