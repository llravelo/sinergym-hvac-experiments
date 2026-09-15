from sinergym.utils.rewards import NormalizedLinearReward


class ProjectReward(NormalizedLinearReward):
    """Project-wide reward function.

    Currently identical to Sinergym's NormalizedLinearReward. Experiment
    configs reference this class rather than NormalizedLinearReward
    directly, so future reward customization happens here without needing
    to touch every config.
    """
