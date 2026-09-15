import importlib

import gymnasium as gym
import sinergym  # noqa: F401  (registers Eplus-* env ids)
from sinergym.utils.wrappers import LoggerWrapper, NormalizeObservation, WandBLogger


def make_env(config: dict) -> gym.Env:
    """Builds a wrapped Sinergym env from an experiment config dict.

    Always applies NormalizeObservation (raw observations span wildly
    different scales/units) and LoggerWrapper (episode-summary metrics).
    Wraps with WandBLogger only if config['wandb']['entity'] is set, so
    configs can be run without wandb credentials configured yet.
    """
    reward_class = _resolve_class(config['reward']['class'])

    env = gym.make(
        config['env_id'],
        reward=reward_class,
        reward_kwargs=config['reward']['kwargs'],
    )
    env = NormalizeObservation(env)
    env = LoggerWrapper(env)

    wandb_cfg = config.get('wandb') or {}
    if wandb_cfg.get('entity') and wandb_cfg.get('project'):
        env = WandBLogger(
            env,
            entity=wandb_cfg['entity'],
            project_name=wandb_cfg['project'],
            group=wandb_cfg.get('group'),
            tags=wandb_cfg.get('tags'),
        )

    return env


def _resolve_class(dotted_path: str):
    module_path, class_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
