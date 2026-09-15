import argparse

import yaml
from stable_baselines3 import DQN

from common.envs import make_env

ALGORITHMS = {
    'DQN': DQN,
}

def main(config_path: str) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = make_env(config)

    algorithm_cls = ALGORITHMS[config['model']['algorithm']]
    model = algorithm_cls(
        config['model']['policy'],
        env,
        seed=config['model']['seed'],
        verbose=1,
        **config['model']['hyperparameters'],
    )

    model.learn(total_timesteps=config['train']['total_timesteps'])

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to a YAML experiment config')
    args = parser.parse_args()
    main(args.config)
