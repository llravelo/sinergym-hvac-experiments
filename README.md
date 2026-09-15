# 43008 Reinforcement Learning — Sinergym setup

Reproducible Docker setup for [Sinergym](https://github.com/ugr-sail/sinergym) (a Gymnasium environment wrapping the EnergyPlus building simulator), plus a hello-world notebook that verifies the simulator runs correctly.

Python, EnergyPlus, and Sinergym's dependencies run inside the container. The host only needs Docker and `make`.

## Prerequisites

- **Docker** with the Compose plugin (`docker compose`, not the older standalone `docker-compose`). Check with:
  ```bash
  docker compose version
  ```
- **`make`**. Check with `make --version`. If missing: `sudo apt install make` (Debian/Ubuntu, often already present via `build-essential`), Xcode Command Line Tools on macOS, or WSL/Git Bash on Windows.

The Docker image includes Sinergym, EnergyPlus, and their Python dependencies pre-installed.

## Setup

```bash
git clone <this-repo-url>
cd 43008-reinforcement-learning-at03
docker compose build
make verify
```

- `docker compose build` builds the Docker image (pulls the pinned Sinergym base image the first time — can take a few minutes; cached after that).
- `make verify` runs the hello-world notebook end-to-end inside a container and confirms Sinergym works. If it finishes with no errors, the environment is working.

Everything runs in **ephemeral containers**: each `make` command starts a throwaway container, runs one thing, and removes the container when it's done. There's never a container left running in the background that you need to remember to stop.

## Makefile commands

| Command | What it does |
|---|---|
| `make verify` | Headlessly runs a notebook top-to-bottom in a throwaway container and saves the executed outputs back into the file — a quick way to confirm a notebook's pipeline still works, without watching it run. Defaults to `notebooks/hello_world.ipynb`; point it at a different notebook with `make verify NB=notebooks/your_notebook.ipynb`. |
| `make train` | Runs a config-driven training script (`scripts/train.py`) in a throwaway container. Defaults to `configs/dqn_5zone_hot.yaml`; point it at a different config with `make train CONFIG=configs/your_config.yaml`. See "Experiment configs" below. |
| `make lab` | Starts an interactive JupyterLab server you can use from your browser, for actually writing/running code cell-by-cell. Prints a URL like `http://localhost:8888/...?token=...` — open that in your browser. Press Ctrl+C in the terminal to stop it; the container is removed automatically. |
| `make shell` | Drops you into an interactive `bash` shell inside a throwaway container — useful for poking around, checking installed packages, or running one-off commands. Type `exit` to leave; the container is removed automatically. |
| `make clean` | Deletes the `Eplus-*-res*/` folders that simulations leave behind (raw EnergyPlus engine output — error logs, sizing calcs, etc; not something you need to keep). Uses the container to do the deleting because those folders are owned by `root` on your host (see note below) and a plain `rm -rf` from your own shell will fail with a permissions error. |

## Experiment configs

`configs/*.yaml` describe a training run: env id, reward class and kwargs, model algorithm and hyperparameters, total timesteps, and optional wandb settings. `scripts/train.py` reads one and runs it (`make train CONFIG=configs/your_config.yaml`). Shared code (the env-building logic, the project's reward class) lives in `common/`.

To enable wandb logging, set `wandb.entity` (and `wandb.project`, already set) in the config, and export `WANDB_API_KEY` in your shell before running `make train` — it's passed through to the container automatically. With `entity` left `null`, training runs without wandb logging.

The project reward class is `common.rewards.ProjectReward` (currently identical to Sinergym's `NormalizedLinearReward`) — edit it there as reward requirements change, rather than in each config.

## Notes

- **Container runs as root.** Anything a simulation writes into this bind-mounted repo (the `Eplus-*-res*/` output folders, the `.lock` file Sinergym uses internally to avoid race conditions when creating those folders) ends up owned by `root` on your host. Both are already gitignored, and `make clean` handles removing the output folders for you — you shouldn't need to touch them directly.
- **These output folders accumulate.** Every simulation run creates a new `Eplus-*-res<N>/` folder rather than overwriting the last one, so disk usage grows the more you run. Run `make clean` periodically, especially during active experimentation — not just before committing.