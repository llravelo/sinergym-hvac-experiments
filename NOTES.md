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

### Cross-model comparison pitfall

Discrete DQN/PPO get a 10-action grid; continuous PPO gets the full 2D continuous space. This is not an apples-to-apples comparison of algorithm quality — continuous PPO has strictly finer control resolution by construction, so part of any performance gap reflects that rather than the RL algorithm itself. Decide upfront whether that's a deliberate part of the evaluation question (discretization cost) or something to control for (e.g., a finer discrete grid).
