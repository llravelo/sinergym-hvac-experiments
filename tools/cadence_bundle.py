"""GUI bundle contract: the only interface between the simulation notebook and the web dashboard.

Both producers use this module: `make_mock_bundle.py` imports it, and notebook/cadence_hvac_dqn_ppo_v2.ipynb embeds a
verbatim copy of everything below the imports, so the notebook stays self-contained (keep the two identical).
Column names and dtypes here must stay in sync with `web/src/bundle/columns.ts`.
See PROPOSED_GUI.md §5 for the contract and §6 for the notebook changes.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

SCHEMA_VERSION: Final[int] = 1

TIME_COLUMNS: Final[tuple[str, ...]] = (
    "step", "decision_idx", "day_of_year", "month", "day_of_month", "hour", "minute", "day_of_week", "holiday_index",
)
RAW_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "outdoor_temperature", "outdoor_humidity", "wind_speed", "wind_direction",
    "diffuse_solar_radiation", "direct_solar_radiation",
    "heating_setpoint", "cooling_setpoint", "air_temperature", "air_humidity",
    "occupancy", "co2_emission", "hvac_power", "hvac_energy", "reward",
)
DERIVED_FLOAT_COLUMNS: Final[tuple[str, ...]] = (
    "comfort_low", "comfort_up", "violation", "hours_until_open", "energy_term", "comfort_term", "energy_kWh",
)
DERIVED_BOOL_COLUMNS: Final[tuple[str, ...]] = ("workday", "operating")

TRACE_DTYPES: Final[dict[str, str]] = {
    **{c: "int32" for c in TIME_COLUMNS},
    **{c: "float32" for c in RAW_FLOAT_COLUMNS},
    "action": "int32",
    **{c: "float32" for c in DERIVED_FLOAT_COLUMNS},
    **{c: "bool" for c in DERIVED_BOOL_COLUMNS},
}

METRIC_COLUMNS: Final[tuple[str, ...]] = (
    "mean_reward", "mean_power_W", "hvac_energy_kWh", "comfort_violation_op_pct", "unoccupied_excursion_pct",
    "mean_violation_op_C", "occupied_violation_pct", "energy_saving_vs_fixed_pct",
)
TRAINING_COLUMNS: Final[tuple[str, ...]] = ("episode", "timesteps", "return", "length", "elapsed_min")

MANIFEST_KEYS: Final[tuple[str, ...]] = (
    "schema_version", "bundle_id", "created_at", "is_mock", "nominal_year", "timesteps_per_hour", "action_repeat",
    "operating_hours", "workdays_eplus", "comfort", "reward", "actions", "train_climate", "climates", "policies",
    "algorithms", "feature_names", "metric_labels",
)

FIXED_POLICY_KEY: Final[str] = "fixed_setpoint"
METRIC_TOLERANCE: Final[float] = 1e-3


class BundleValidationError(ValueError):
    pass


def policy_key(name: str) -> str:
    """Same key as the notebook's `evaluate_or_load` file names (e.g. 'Rule-based' -> 'rule_based')."""
    return name.lower().replace(" ", "_").replace("-", "_")


def score_columns(n_actions: int) -> list[str]:
    return [f"score_{i}" for i in range(n_actions)]


def feature_columns(feature_names: Sequence[str]) -> list[str]:
    return [f"f_{name}" for name in feature_names]


def decision_dtypes(feature_names: Sequence[str], n_actions: int) -> dict[str, str]:
    """`f_<feature>` columns are unused by the dashboard but kept for the report's decision analysis (PROPOSED_GUI.md §12)."""
    return {
        "decision_idx": "int32",
        "action": "int32",
        **{c: "float32" for c in feature_columns(feature_names)},
        **{c: "float32" for c in score_columns(n_actions)},
        "value": "float32",
    }


def nominal_year(first_day_of_week: int) -> int:
    """A non-leap year whose 1 January falls on `first_day_of_week` (EnergyPlus: 1 = Sunday … 7 = Saturday)."""
    for year in range(2001, 2029):
        is_leap = year % 4 == 0
        eplus_day_of_week = (date(year, 1, 1).weekday() + 1) % 7 + 1
        if not is_leap and eplus_day_of_week == first_day_of_week:
            return year
    raise ValueError(f"Invalid EnergyPlus day of week: {first_day_of_week}")


def build_manifest(
    *,
    bundle_id: str,
    is_mock: bool,
    first_day_of_week: int,
    timesteps_per_hour: int,
    action_repeat: int,
    operating_hours: tuple[int, int],
    workdays: Sequence[int],
    comfort_winter: tuple[float, float],
    comfort_summer: tuple[float, float],
    summer_start: tuple[int, int],
    summer_end: tuple[int, int],
    reward: Mapping[str, float],
    action_setpoints: Sequence[tuple[float, float]],
    train_climate: str,
    climates: Sequence[Mapping[str, str]],
    policies: Sequence[Mapping[str, Any]],
    algorithms: Mapping[str, Mapping[str, Any]],
    feature_names: Sequence[str],
    metric_labels: Mapping[str, str],
) -> dict[str, Any]:
    """Serialise the notebook constants the GUI needs. `policies` items: key, name, kind, score_type, color."""
    return {
        "schema_version": SCHEMA_VERSION,
        "bundle_id": bundle_id,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "is_mock": is_mock,
        "nominal_year": nominal_year(first_day_of_week),
        "timesteps_per_hour": timesteps_per_hour,
        "action_repeat": action_repeat,
        "operating_hours": list(operating_hours),
        "workdays_eplus": sorted(workdays),
        "comfort": {
            "winter": list(comfort_winter),
            "summer": list(comfort_summer),
            "summer_start": list(summer_start),
            "summer_end": list(summer_end),
        },
        "reward": dict(reward),
        "actions": [
            {"id": i, "heating": heating, "cooling": cooling}
            for i, (heating, cooling) in enumerate(action_setpoints)
        ],
        "train_climate": train_climate,
        "climates": [dict(c) for c in climates],
        "policies": [dict(p) for p in policies],
        "algorithms": {key: dict(value) for key, value in algorithms.items()},
        "feature_names": list(feature_names),
        "metric_labels": dict(metric_labels),
    }


def trace_metrics(trace: pd.DataFrame, timesteps_per_hour: int) -> dict[str, float]:
    """Notebook `compute_metrics`, computed from the derived columns of a bundle trace."""
    outside = trace["violation"].to_numpy(dtype=float) > 0
    operating = trace["operating"].to_numpy(dtype=bool)
    occupied = trace["occupancy"].to_numpy(dtype=float) > 0
    power = trace["hvac_power"].to_numpy(dtype=float)
    violation = trace["violation"].to_numpy(dtype=float)
    return {
        "mean_reward": float(trace["reward"].to_numpy(dtype=float).mean()),
        "mean_power_W": float(power.mean()),
        "hvac_energy_kWh": float(power.sum() / timesteps_per_hour / 1000),
        "comfort_violation_op_pct": float(100 * outside[operating].mean()),
        "unoccupied_excursion_pct": float(100 * outside[~operating].mean()),
        "mean_violation_op_C": float(violation[operating].mean()),
        "occupied_violation_pct": float(100 * outside[occupied].mean()) if occupied.any() else float("nan"),
    }


def add_energy_saving(metrics: pd.DataFrame) -> pd.DataFrame:
    """Adds `energy_saving_vs_fixed_pct` (positive = less energy than the fixed setpoint in the same climate)."""
    metrics = metrics.copy()
    fixed = metrics[metrics["policy"] == FIXED_POLICY_KEY].set_index("climate")["hvac_energy_kWh"]
    metrics["energy_saving_vs_fixed_pct"] = 100 * (1 - metrics["hvac_energy_kWh"] / metrics["climate"].map(fixed))
    return metrics


def write_bundle(
    out_dir: Path,
    manifest: Mapping[str, Any],
    traces: Mapping[tuple[str, str], pd.DataFrame],
    decisions: Mapping[tuple[str, str], pd.DataFrame],
    metrics: pd.DataFrame,
    training: Mapping[str, pd.DataFrame],
) -> Path:
    """Writes a bundle. Keys of `traces`/`decisions` are (climate, policy_key); `decisions` may omit entries.

    Extra trace columns (e.g. perimeter zone temperatures) are kept as float32, so optional data needs no schema change.
    """
    out_dir = Path(out_dir)
    _prepare_output_dir(out_dir)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))

    for (climate, policy), trace in traces.items():
        _write_table(out_dir / "traces" / climate / f"{policy}.parquet", _cast(trace, TRACE_DTYPES, keep_extra=True))
    dtypes = decision_dtypes(manifest["feature_names"], len(manifest["actions"]))
    for (climate, policy), table in decisions.items():
        _write_table(out_dir / "decisions" / climate / f"{policy}.parquet", _cast(table, dtypes, keep_extra=False))

    metric_dtypes = {"policy": "string", "climate": "string", **{c: "float32" for c in METRIC_COLUMNS}}
    _write_table(out_dir / "metrics.parquet", _cast(metrics, metric_dtypes, keep_extra=False))
    for algo, log in training.items():
        log_dtypes = {**{c: "float32" for c in log.columns}, "episode": "int32", "timesteps": "int32", "length": "int32"}
        _write_table(out_dir / "training" / f"{algo}.parquet", _cast(log, log_dtypes, keep_extra=False))
    return out_dir


def validate_bundle(bundle_dir: Path) -> None:
    """Checks the bundle is complete and self-consistent. Raises BundleValidationError listing every problem."""
    bundle_dir = Path(bundle_dir)
    issues: list[str] = []
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise BundleValidationError(f"{manifest_path} is missing")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())

    missing_keys = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing_keys:
        raise BundleValidationError(f"manifest.json is missing keys: {missing_keys}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise BundleValidationError(f"schema_version {manifest['schema_version']} != {SCHEMA_VERSION}")

    repeat: int = manifest["action_repeat"]
    n_actions: int = len(manifest["actions"])
    climates = [c["key"] for c in manifest["climates"]]
    policies = {p["key"]: p for p in manifest["policies"]}
    computed_metrics: dict[tuple[str, str], dict[str, float]] = {}

    for climate in climates:
        reference_time: pd.DataFrame | None = None
        for key, policy in policies.items():
            label = f"{climate}/{key}"
            trace = _read_optional(bundle_dir / "traces" / climate / f"{key}.parquet")
            if trace is None:
                issues.append(f"traces/{label}.parquet is missing")
                continue
            missing = sorted(set(TRACE_DTYPES) - set(trace.columns))
            if missing:
                issues.append(f"traces/{label}: missing columns {missing}")
                continue
            if len(trace) % repeat != 0:
                issues.append(f"traces/{label}: {len(trace)} rows is not a multiple of action_repeat={repeat}")
            if not np.array_equal(trace["step"].to_numpy(), np.arange(len(trace))):
                issues.append(f"traces/{label}: 'step' is not 0..n-1")
            if not np.array_equal(trace["decision_idx"].to_numpy(), trace["step"].to_numpy() // repeat):
                issues.append(f"traces/{label}: 'decision_idx' != step // action_repeat")

            time = trace[list(TIME_COLUMNS)]
            if reference_time is None:
                reference_time = time
            elif not time.equals(reference_time):
                issues.append(f"traces/{label}: time columns differ from the other policies in '{climate}'")

            computed_metrics[(key, climate)] = trace_metrics(trace, manifest["timesteps_per_hour"])
            issues.extend(_validate_decisions(bundle_dir, label, trace, policy, manifest["feature_names"], n_actions, repeat))

    issues.extend(_validate_metrics(bundle_dir, computed_metrics, policies, climates))
    for algo in manifest["algorithms"]:
        log = _read_optional(bundle_dir / "training" / f"{algo}.parquet")
        if log is None:
            issues.append(f"training/{algo}.parquet is missing")
        elif missing := sorted(set(TRAINING_COLUMNS) - set(log.columns)):
            issues.append(f"training/{algo}: missing columns {missing}")

    if issues:
        raise BundleValidationError("Bundle validation failed:\n  - " + "\n  - ".join(issues))


def zip_bundle(bundle_dir: Path) -> Path:
    bundle_dir = Path(bundle_dir)
    return Path(shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir))


def _validate_decisions(
    bundle_dir: Path, label: str, trace: pd.DataFrame, policy: Mapping[str, Any], feature_names: Sequence[str],
    n_actions: int, repeat: int,
) -> list[str]:
    table = _read_optional(bundle_dir / "decisions" / f"{label}.parquet")
    if table is None:
        return []  # optional: the GUI hides the agent explanation
    missing = sorted(set(decision_dtypes(feature_names, n_actions)) - set(table.columns))
    if missing:
        return [f"decisions/{label}: missing columns {missing}"]
    issues: list[str] = []
    if len(table) * repeat != len(trace):
        issues.append(f"decisions/{label}: {len(table)} rows, expected {len(trace) // repeat}")
        return issues
    if not np.array_equal(table["decision_idx"].to_numpy(), np.arange(len(table))):
        issues.append(f"decisions/{label}: 'decision_idx' is not 0..n-1")
    if not np.array_equal(table["action"].to_numpy(), trace["action"].to_numpy()[::repeat]):
        issues.append(f"decisions/{label}: actions differ from the trace")
    if policy.get("score_type") is not None:
        scores = table[score_columns(n_actions)].to_numpy(dtype=float)
        if np.isnan(scores).any():
            issues.append(f"decisions/{label}: agent policy has missing scores")
        elif not np.array_equal(scores.argmax(axis=1), table["action"].to_numpy()):
            issues.append(f"decisions/{label}: argmax(score) != action")
    return issues


def _validate_metrics(
    bundle_dir: Path, computed: Mapping[tuple[str, str], dict[str, float]], policies: Mapping[str, Any],
    climates: Sequence[str],
) -> list[str]:
    metrics = _read_optional(bundle_dir / "metrics.parquet")
    if metrics is None:
        return ["metrics.parquet is missing"]
    missing = sorted({"policy", "climate", *METRIC_COLUMNS} - set(metrics.columns))
    if missing:
        return [f"metrics: missing columns {missing}"]
    issues: list[str] = []
    stored = metrics.set_index(["policy", "climate"])
    for key in policies:
        for climate in climates:
            if (key, climate) not in stored.index:
                issues.append(f"metrics: no row for {key}/{climate}")
                continue
            for name, value in computed.get((key, climate), {}).items():
                if not np.isclose(stored.loc[(key, climate), name], value, rtol=METRIC_TOLERANCE, equal_nan=True):
                    issues.append(
                        f"metrics: {key}/{climate} {name}={stored.loc[(key, climate), name]:.4g} "
                        f"but the trace gives {value:.4g}"
                    )
    return issues


def _prepare_output_dir(out_dir: Path) -> None:
    """Replaces a previous bundle in place, but never deletes a folder that isn't a bundle."""
    if out_dir.exists():
        if not (out_dir / "manifest.json").exists() and any(out_dir.iterdir()):
            raise FileExistsError(f"{out_dir} exists and is not a GUI bundle; refusing to overwrite it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)


def _cast(table: pd.DataFrame, dtypes: Mapping[str, str], keep_extra: bool) -> pd.DataFrame:
    missing = sorted(set(dtypes) - set(table.columns))
    if missing:
        raise KeyError(f"Missing columns {missing}")
    extra = [c for c in table.columns if c not in dtypes] if keep_extra else []
    cast = table[list(dtypes)].astype(dict(dtypes))
    for column in extra:
        cast[column] = table[column].astype("float32")
    return cast.reset_index(drop=True)


def _write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False, compression="snappy")


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None
