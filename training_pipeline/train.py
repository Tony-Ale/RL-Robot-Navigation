import argparse
from pathlib import Path

from training_pipeline.utils import (
    configure_matplotlib_cache,
    save_json,
)

configure_matplotlib_cache()

from training_pipeline.architecture_extractor import ArchitectureFeaturesExtractor, effective_robot_input_dim
from training_pipeline.callbacks import NavigationEvaluationCallback
from training_pipeline.env_factory import make_eval_env
from training_pipeline.training_runtime import (
    build_navigation_callbacks,
    initialize_training_run,
    learn_model,
    managed_training_environment,
)
from testing_pipeline.runner import run_testing


def build_model(config, env, run_dir: Path):
    """Build or resume a Stable-Baselines3 PPO model."""
    from stable_baselines3 import PPO

    architecture_cfg = config["architecture"]
    ppo_cfg = config["ppo"]
    policy_kwargs = {
        "features_extractor_class": ArchitectureFeaturesExtractor,
        "features_extractor_kwargs": {
            "architecture_name": architecture_cfg["name"],
            "architecture_config_path": architecture_cfg["config_path"],
            "entity_keys": architecture_cfg.get("entity_keys", ["humans"]),
            "entity_feature_dim": architecture_cfg.get("entity_feature_dim", 14),
            "mask_zero_entities": architecture_cfg.get("mask_zero_entities", True),
            "include_waypoint_features": architecture_cfg.get("include_waypoint_features", False),
        },
    }
    if ppo_cfg.get("policy_net_arch") is not None:
        policy_kwargs["net_arch"] = ppo_cfg["policy_net_arch"]

    checkpoint = config["training"].get("resume_from_checkpoint")
    if checkpoint:
        return PPO.load(
            checkpoint,
            env=env,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=ppo_cfg.get("device", "auto"),
            seed=config["experiment"]["seed"],
            learning_rate=ppo_cfg["learning_rate"],
            n_epochs=ppo_cfg["n_epochs"],
            target_kl=ppo_cfg.get("target_kl"),
        )

    return PPO(
        policy=ppo_cfg.get("policy", "MultiInputPolicy"),
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        target_kl=ppo_cfg.get("target_kl"),
        tensorboard_log=str(run_dir / "tensorboard"),
        policy_kwargs=policy_kwargs,
        device=ppo_cfg.get("device", "auto"),
        verbose=ppo_cfg.get("verbose", 1),
        seed=config["experiment"]["seed"],
    )


def build_callbacks(config, run_dir: Path):
    """Build the callback stack for checkpoints, ETA, CSV metrics, and evaluation."""
    return build_navigation_callbacks(
        config,
        run_dir,
        evaluation_callback_class=NavigationEvaluationCallback,
        eval_env_factory=make_eval_env,
    )


def print_architecture_startup(config, model) -> None:
    """Print the architecture actually attached to the PPO policy."""
    architecture_cfg = config["architecture"]
    extractor = getattr(getattr(model, "policy", None), "features_extractor", None)
    print("Architecture loaded:")
    print(f"  name: {architecture_cfg['name']}")
    print(f"  config: {architecture_cfg['config_path']}")
    print(f"  PPO training envs: {model.n_envs}")
    if extractor is not None:
        print(f"  effective robot input dim: {extractor.effective_robot_input_dim}")
        print(f"  base robot input dim: {extractor.base_robot_input_dim}")
        print(f"  waypoint input dim: {extractor.waypoint_input_dim}")
        print(f"  PPO feature dim: {extractor.features_dim}")
        print(f"  entity keys: {', '.join(extractor.entity_keys)}")


def validate_architecture_entity_keys(config, observation_space) -> None:
    """Fail early if configured entity keys are not visible to PPO."""
    entity_keys = tuple(config["architecture"].get("entity_keys", ["humans"]))
    if not entity_keys or not hasattr(observation_space, "spaces"):
        return

    available_keys = set(observation_space.spaces.keys())
    missing_keys = [key for key in entity_keys if key not in available_keys]
    if missing_keys:
        raise ValueError(
            "architecture.entity_keys contains keys missing from the final wrapped observation space: "
            f"{missing_keys}. Available keys: {sorted(available_keys)}"
        )


def train(config_path: str):
    config, run_dir, env = initialize_training_run(config_path)
    with managed_training_environment(env):
        validate_architecture_entity_keys(config, env.observation_space)
        architecture_cfg = config["architecture"]
        base_robot_dim = effective_robot_input_dim(env.observation_space, False)
        resolved_robot_dim = effective_robot_input_dim(
            env.observation_space,
            architecture_cfg.get("include_waypoint_features", False),
        )
        config["resolved_architecture"] = {
            "effective_robot_input_dim": resolved_robot_dim,
            "base_robot_input_dim": base_robot_dim,
            "waypoint_input_dim": resolved_robot_dim - base_robot_dim,
        }
        save_json(run_dir / "resolved_config.json", config)

        model = build_model(config, env, run_dir)
        print_architecture_startup(config, model)
        callbacks = build_callbacks(config, run_dir)
        learn_model(config, model, callbacks)
    testing_cfg = config.get("testing", {})
    if testing_cfg.get("enabled", False) and testing_cfg.get("run_after_training", False):
        checkpoint_path = run_dir / "checkpoints" / f"ppo_final_step_{model.num_timesteps}.zip"
        run_testing(config, run_dir, checkpoint_path=checkpoint_path)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Train PPO on SocNavGym with a selected architecture.")
    parser.add_argument("--config", default="training_pipeline/config.yaml", help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_dir = train(args.config)
    print(f"Training complete. Run data saved to: {run_dir}")


if __name__ == "__main__":
    main()
