# Project notes

Working notes on the plan toward running and evaluating RL models (discrete DQN, discrete PPO, continuous PPO) on Sinergym, and known pitfalls specific to this setup.

## High-level plan

1. **Folder structure — done.** Shared experiment config schema (`configs/*.yaml`), a reward module (`common/rewards.py`), and a config-driven run script (`scripts/train.py`) instead of one notebook per run. Notebooks are for exploration and analysis of results, not for launching training runs — with three models times multiple reward variants times hyperparameter sweeps, hand-editing notebook cells per run doesn't scale and makes runs hard to reproduce or compare. Evaluation harness not yet built — deferred until baseline runs (step 4).
2. **wandb integration — done.** `common/envs.py` wraps envs with `WandBLogger` (Sinergym's own wrapper, requires `LoggerWrapper` applied first) when a config's `wandb.entity`/`wandb.project` are set; otherwise training runs without it. `WANDB_API_KEY` is passed through from the host shell into the container via `docker-compose.yml`'s `environment:` block.
3. **Reward calibration — in progress.** Currently using `NormalizedLinearReward` (via `common.rewards.ProjectReward`, a thin subclass kept as the one place to customize the reward as requirements change, so configs don't need to). Sinergym also ships `LinearReward`, `HourlyLinearReward`, `ExpReward`, `EnergyCostLinearReward`, `MultiZoneReward` (`sinergym.utils.rewards`) — `HourlyLinearReward` already implements a time-of-day-dependent comfort/energy weighting (`range_comfort_hours`, default energy weight outside those hours), worth comparing against. `NormalizedLinearReward` solves the energy-vs-comfort magnitude mismatch by normalizing both penalty terms against running max values instead of hand-tuned scale constants. This phase is selecting and tuning kwargs on existing reward classes, comparing reward-term breakdowns (`energy_term`/`comfort_term`) across settings in wandb — not writing a custom reward from scratch, unless the built-ins turn out to be insufficient.
4. **Baseline runs — deferred, will need customization.** Sinergym ships `RBC5Zone` (rule-based controller for this building) and `RandomController` (`sinergym.utils.controllers`). `RandomController` is fine to use as-is. `RBC5Zone` is a pre-tuned rule-based controller — using it unmodified isn't a fair baseline for demonstrating RL's value over hand-tuning, so it needs to be de-tuned/simplified before use as a comparison point. Run both under the final, settled reward function before spending compute on model tuning.
5. **Hyperparameter experiments** per model (discrete DQN, discrete PPO, continuous PPO), compared against the baseline from step 4.

## Pitfalls

Checked against the installed Sinergym version (3.12.2) and stable-baselines3 (2.9.0) source, not general assumption.

### Shared across all three models

- **Observation scale.** The declared observation space is `Box(-5e7, 5e7, (17,), float32)` — a generic wide placeholder, not the real value range. Actual values span very different units in the same vector (temperature ~20, CO2 mass ~1e7, hour-of-day 0–23, power ~1e4). A default MLP policy trained on raw, unnormalized observations like this will train poorly or not at all. Sinergym's `NormalizeObservation` wrapper (running mean/std, auto-calibrating across episodes) should be applied before training, for all three models.
- **Episode length vs. default hyperparameters.** One episode is 35,040 steps (a simulated year at 15-minute resolution), and each step is a real EnergyPlus computation, not a cheap simulator step. SB3's DQN defaults (`buffer_size=1_000_000`, `learning_starts=50_000`) assume cheap-to-sample environments; `learning_starts=50000` alone could take a long time wall-clock here. These need deliberate overriding rather than being left at defaults.
- **Throughput/parallelism.** Sample collection is CPU-bound and single-process by default. Vectorizing (`SubprocVecEnv`) speeds up wall-clock for rollout collection, but each parallel env is a full separate EnergyPlus process — CPU/RAM usage scales with `n_envs` and needs deliberate resource budgeting.
- **Reproducibility across model comparisons.** Three separate seeds are in play: SB3's algorithm seed, the env's `reset(seed=...)`, and the weather-noise seed (`-stochastic-` env variants). Fair comparison across the three models requires all three controlled and logged per run.
- **Cheap sanity check:** run `stable_baselines3.common.env_checker.check_env` before training to catch space/dtype mismatches early.

### Discrete-specific (DQN + discrete PPO)

- **Discretization granularity is a real design choice.** `Eplus-5zone-hot-discrete-v1` wraps the real 2D `Box([12, 23.25], [23.25, 30])` action space down to `Discrete(10)` via the `DiscretizeEnv` wrapper (arbitrary `action_mapping: Callable[[int], np.ndarray]`). 10 actions over a 2D continuous control problem is coarse and directly caps the best achievable comfort/energy tradeoff, independent of how well the algorithm trains. Worth inspecting what the 10 mapped setpoint pairs actually are before attributing a poor result to the algorithm rather than the discretization.
- **DQN strictly requires a `Discrete` action space** (not `MultiDiscrete`/`Box`) — a non-issue with these env IDs, but breaks immediately if DQN is ever pointed at a continuous env.
- **Discrete PPO uses a Categorical policy head**, not Gaussian — exploration behavior differs enough from continuous PPO that hyperparameters shouldn't be copy-pasted between the two "PPO" runs; they're the same algorithm class but effectively separate tuning problems.

### Continuous-specific (PPO)

- **Action space isn't centered near zero.** Real bounds are heating `[12, 23.25]`, cooling `[23.25, 30]` — both far from 0, asymmetric. PPO's Gaussian policy head initializes to output near 0. Sinergym's `NormalizeAction` wrapper (rescales the real Box to `[-1, 1]` for the policy, maps back before sending to the simulator) exists specifically for this — its docstring calls it "very useful in DRL algorithms." Skipping it risks a policy whose initial actions saturate against the real bounds, slowing early learning.
- **Two coupled actuators.** Heating and cooling setpoints are sampled independently by a diagonal Gaussian; nothing in the action space itself prevents sampling a physically backwards pair (heating setpoint above cooling setpoint). Worth checking whether Sinergym clips/handles this internally or leaves it for the agent to learn to avoid.

### Config schema convention: shared vs. model-specific fields

When writing a new model config (e.g. the discrete/continuous PPO configs), decide where each field goes by asking: does it define *what problem* is being solved, or *how a particular algorithm* solves it?

- **Shared — must be identical across every config in the comparison:** `reward.class`/`reward.kwargs`, `model.seed`, `model.gamma` (defines the discounted-return objective itself, not algorithm machinery), `train.total_timesteps` (the sample budget each algorithm gets). Also enforced in code rather than YAML: the `NormalizeObservation`/`LoggerWrapper` wrapper stack in `common/envs.py` applies to every config the same way.
- **The independent variable — deliberately *not* forced equal:** `env_id`. DQN and discrete PPO should use the *same* `env_id` (that pairing is the real algorithm-quality comparison); continuous PPO necessarily uses a different one. This is the thing being studied, not a hyperparameter to tune for convenience.
- **Model-specific — free to differ, tune per algorithm:** everything under `model.hyperparameters` (DQN's `buffer_size`/`learning_starts`/`train_freq`/`target_update_interval` have no PPO analog; PPO's `n_steps`/`n_epochs`/`gae_lambda`/`ent_coef`/`clip_range` have no DQN analog), `learning_rate` (value-based and policy-gradient losses live on different scales — matching the number isn't "more fair," it's arbitrary), network architecture (`policy_kwargs`), and the exploration mechanism (DQN's epsilon-greedy schedule vs. PPO's entropy bonus). Note that same-named fields (e.g. `batch_size`) don't need matching values across algorithms since they play different mechanical roles.
- `wandb.group`/`wandb.tags` should *differ* per config (that's how runs get told apart); `wandb.entity`/`wandb.project` should be *shared* team-wide so everyone's runs land in one place to compare.
- `eval.*` (below) is also shared: evaluating one model over fewer/more episodes than another biases the comparison.

### Evaluation

Two-tier, following Manjavacas et al. 2024's methodology (their freq=4 training episodes / eval length=3 / final eval=20 episodes), adapted for our brief's extra metric:

1. **Periodic in-training eval** (`scripts/train.py`, SB3 `EvalCallback`, config's `eval.freq_timesteps`/`eval.n_episodes`) -- runs on a separate eval env instance (never `WandBLogger`-wrapped, to avoid double `wandb.init()` in one process), `deterministic=True`, saves the best checkpoint to `models/<wandb.group>/best_model.zip` rather than trusting whatever the final timestep happens to land on.
2. **Final held-out evaluation** (`scripts/evaluate.py`, `python -m scripts.evaluate <config> <model_path>`) -- runs `eval.final_n_episodes` episodes with the best checkpoint and reports mean±SD for mean reward, mean power demand, and comfort violation split into **operational vs. unoccupied hours** (the brief's metric, not in the paper) using per-step occupancy.

Even a fully deterministic policy (SB3 DQN's default eval mode, no epsilon) on a non-stochastic weather file isn't guaranteed to produce identical episodes -- the paper's own RBC baseline (a fixed rule, no learning) shows nonzero variance across 20 "identical" eval episodes (their Table 9), implying EnergyPlus itself has some run-to-run noise. Report mean±SD across multiple eval episodes for every controller, including DQN, rather than assuming 1 episode suffices because the policy is deterministic.

**Outstanding verification (blocks trusting `scripts/evaluate.py`'s numbers):** `_extract_step_metrics()` in that file uses placeholder `info` dict keys (`total_power_demand`, `total_temperature_violation`, `occupant_count`) that are NOT confirmed against the installed Sinergym version. Run this in `make shell` before relying on eval output:

```
python -c "
import gymnasium as gym, sinergym
env = gym.make('Eplus-5zone-hot-discrete-v1')
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(sorted(info.keys()))
print(info)
try:
    print(env.get_wrapper_attr('observation_variables'))
except Exception as e:
    print('no observation_variables:', e)
"
```

and fix the keys in `_extract_step_metrics` to match what's actually returned.

**Second outstanding verification: `NormalizeObservation` statistics aren't synced between the training env and the eval env.** `NormalizeObservation` maintains a *running* mean/variance and whitens observations against it -- it's applied identically (same procedure) to every config, but the concrete numbers it produces are data-dependent per env instance, since the running stats are estimated online from whatever trajectory that instance has seen. `scripts/train.py`'s `eval_env` gets its own fresh, independently-initialized `NormalizeObservation` -- not synced with the training env's accumulated stats -- so `EvalCallback` may be scoring checkpoints under a materially different observation distribution than the one the policy was actually trained on at that point (worse early in training, before stats converge). SB3 has `stable_baselines3.common.vec_env.sync_envs_normalization` for exactly this problem with its own `VecNormalize`, but that's built for `VecEnv`-wrapped `VecNormalize`, not necessarily compatible with Sinergym's `NormalizeObservation` on a plain `gym.Env`. Before trusting `EvalCallback`'s checkpoint selection, check inside the container whether `NormalizeObservation` exposes anything to sync/freeze/copy its stats (e.g. `env.get_wrapper_attr('mean')`/`var`, a `.training` flag, a save/load method) -- inspect the class directly:

```
python -c "
import gymnasium as gym, sinergym, inspect
from sinergym.utils.wrappers import NormalizeObservation
print(inspect.getsource(NormalizeObservation))
"
```

### Cross-model comparison pitfall

Discrete DQN/PPO get a 10-action grid; continuous PPO gets the full 2D continuous space. This is not an apples-to-apples comparison of algorithm quality — continuous PPO has strictly finer control resolution by construction, so part of any performance gap reflects that rather than the RL algorithm itself. Decide upfront whether that's a deliberate part of the evaluation question (discretization cost) or something to control for (e.g., a finer discrete grid).
