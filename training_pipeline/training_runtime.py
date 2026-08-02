"""Shared orchestration for ordinary and stateful training entry points."""

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Optional

from training_pipeline.callbacks import NavigationTrainingCallback, TrainingRenderCallback
from training_pipeline.env_factory import make_vec_env
from training_pipeline.utils import load_yaml, make_run_dir, record_training_seed_session, set_global_seeds


def initialize_training_run(config_path: str, validate: Optional[Callable[[Dict], None]] = None):
    """Load configuration and create the run directory and vector environment once."""
    config = load_yaml(config_path)
    if validate is not None:
        validate(config)
    set_global_seeds(int(config["experiment"]["seed"]))
    run_dir = make_run_dir(config, config_path)
    record_training_seed_session(run_dir, config)
    return config, run_dir, make_vec_env(config)


def build_navigation_callbacks(
    config: Dict,
    run_dir: Path,
    evaluation_callback_class,
    eval_env_factory,
):
    """Build the callback stack shared by ordinary and recurrent PPO training."""
    from stable_baselines3.common.callbacks import CallbackList

    training = config["training"]
    metrics = config["metrics"]
    callbacks = [
        NavigationTrainingCallback(
            run_dir=run_dir,
            total_timesteps=training["total_timesteps"],
            checkpoint_interval_steps=training["checkpoint_interval_steps"],
            eta_log_interval_steps=metrics["eta_log_interval_steps"],
            navigation_csv_name=metrics["navigation_training_csv"],
            reset_num_timesteps=training.get("reset_num_timesteps", True),
            verbose=config["ppo"].get("verbose", 1),
        )
    ]

    environment = config.get("environment", {})
    if environment.get("render_during_training", False):
        callbacks.append(TrainingRenderCallback(environment.get("render_interval_steps", 1)))

    evaluation = config["evaluation"]
    if evaluation.get("enabled", True):
        callbacks.append(
            evaluation_callback_class(
                eval_env=eval_env_factory(config),
                run_dir=run_dir,
                eval_interval_steps=evaluation["eval_interval_steps"],
                n_eval_episodes=evaluation["n_eval_episodes"],
                deterministic=evaluation["deterministic"],
                navigation_csv_name=metrics["navigation_evaluation_csv"],
                fixed_episode_seeds=evaluation.get("fixed_episode_seeds", True),
                eval_seed_base=evaluation["eval_seed_base"],
                verbose=config["ppo"].get("verbose", 1),
            )
        )
    return CallbackList(callbacks)


@contextmanager
def managed_training_environment(env):
    """Keep environment cleanup reliable across model setup and learning failures."""
    try:
        yield env
    finally:
        env.close()


def learn_model(config: Dict, model, callbacks) -> None:
    """Run one training session using the pipeline's common learning arguments."""
    model.learn(
        total_timesteps=config["training"]["total_timesteps"],
        callback=callbacks,
        log_interval=config["training"]["log_interval"],
        reset_num_timesteps=config["training"].get("reset_num_timesteps", True),
        tb_log_name=config["experiment"]["name"],
    )
