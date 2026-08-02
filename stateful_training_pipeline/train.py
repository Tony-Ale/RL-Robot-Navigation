import argparse
from pathlib import Path
from typing import Any, Dict

from training_pipeline.env_factory import make_eval_env
from training_pipeline.training_runtime import (
    build_navigation_callbacks,
    initialize_training_run,
    learn_model,
    managed_training_environment,
)
from training_pipeline.utils import load_yaml, save_json

from stateful_training_pipeline.callbacks import RecurrentNavigationEvaluationCallback
from stateful_training_pipeline.policy import StatefulSocialContextPolicy
from stateful_training_pipeline.recurrent_ppo import StatefulSocialRecurrentPPO


STABLE_NATIVE_ENTITY_KEYS = frozenset(("humans", "laptops", "tables", "plants"))


def validate_config(config: Dict[str, Any]) -> None:
    """Reject configurations that violate recurrent-state assumptions."""
    if config.get("wrappers", {}).get("observation_history", {}).get("enabled", False):
        raise ValueError("Stateful training requires wrappers.observation_history.enabled: false.")
    if config["architecture"].get("name") != "stateful_social_context_fusion":
        raise ValueError("The stateful pipeline requires architecture.name: stateful_social_context_fusion.")

    entity_keys = tuple(config["architecture"].get("entity_keys", ("humans",)))
    if not entity_keys:
        raise ValueError("Stateful social context fusion requires at least one entity key.")
    if len(entity_keys) != len(set(entity_keys)):
        raise ValueError("Stateful architecture.entity_keys must not contain duplicates.")

    wall_config = config.get("wrappers", {}).get("nearest_wall_segments", {})
    wall_key = wall_config.get("observation_key", "walls")
    supported_keys = STABLE_NATIVE_ENTITY_KEYS | {wall_key}
    unsupported_keys = sorted(set(entity_keys) - supported_keys)
    if unsupported_keys:
        raise ValueError(
            "Stateful architecture.entity_keys contains unsupported or slot-unstable keys: "
            f"{unsupported_keys}. Supported keys: {sorted(supported_keys)}."
        )
    if wall_key in entity_keys:
        if not wall_config.get("enabled", False):
            raise ValueError(
                f"Stateful wall entity key '{wall_key}' requires nearest_wall_segments.enabled: true."
            )
        if wall_config.get("mode", "nearest") != "all":
            raise ValueError(
                "Stateful wall memory requires nearest_wall_segments.mode: all; "
                "distance-ranked wall slots can change identity between steps."
            )

    environment_config = load_yaml(config["environment"]["config_path"])
    env_values = environment_config.get("env", {})
    if env_values.get("get_padded_observations") is not True:
        raise ValueError("Stateful per-entity memory requires get_padded_observations: true.")
    interaction_keys = (
        "max_h_h_dynamic_interactions",
        "max_h_h_dynamic_interactions_non_dispersing",
        "max_h_h_static_interactions",
        "max_h_h_static_interactions_non_dispersing",
        "max_h_l_interactions",
        "max_h_l_interactions_non_dispersing",
        "crowd_formation_probability",
        "crowd_dispersal_probability",
        "human_laptop_formation_probability",
        "human_laptop_dispersal_probability",
    )
    enabled_interactions = [key for key in interaction_keys if float(env_values.get(key, 0)) != 0]
    if enabled_interactions:
        raise ValueError(
            "Stateful per-entity memory requires stable slots; disable interactions. "
            f"Non-zero settings: {enabled_interactions}."
        )


def build_model(config: Dict[str, Any], env, run_dir: Path) -> StatefulSocialRecurrentPPO:
    architecture = config["architecture"]
    ppo = config["ppo"]
    policy_kwargs = {
        "architecture_config_path": architecture["config_path"],
        "entity_keys": architecture.get("entity_keys", ["humans"]),
        "entity_feature_dim": architecture.get("entity_feature_dim", 14),
        "mask_zero_entities": architecture.get("mask_zero_entities", True),
        "include_waypoint_features": architecture.get("include_waypoint_features", False),
        "net_arch": ppo.get("policy_net_arch", []),
    }
    checkpoint = config["training"].get("resume_from_checkpoint")
    if checkpoint:
        return StatefulSocialRecurrentPPO.load(
            checkpoint,
            env=env,
            tensorboard_log=str(run_dir / "tensorboard"),
            device=ppo.get("device", "auto"),
            seed=config["experiment"]["seed"],
            learning_rate=ppo["learning_rate"],
            n_epochs=ppo["n_epochs"],
            target_kl=ppo.get("target_kl"),
        )

    return StatefulSocialRecurrentPPO(
        policy=StatefulSocialContextPolicy,
        env=env,
        learning_rate=ppo["learning_rate"],
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"],
        ent_coef=ppo["ent_coef"],
        vf_coef=ppo["vf_coef"],
        max_grad_norm=ppo["max_grad_norm"],
        target_kl=ppo.get("target_kl"),
        tensorboard_log=str(run_dir / "tensorboard"),
        policy_kwargs=policy_kwargs,
        device=ppo.get("device", "auto"),
        verbose=ppo.get("verbose", 1),
        seed=config["experiment"]["seed"],
    )


def build_callbacks(config: Dict[str, Any], run_dir: Path):
    return build_navigation_callbacks(
        config,
        run_dir,
        evaluation_callback_class=RecurrentNavigationEvaluationCallback,
        eval_env_factory=make_eval_env,
    )


def train(config_path: str) -> Path:
    config, run_dir, env = initialize_training_run(config_path, validate=validate_config)
    with managed_training_environment(env):
        model = build_model(config, env, run_dir)
        config["resolved_stateful_architecture"] = {
            "recurrent_state_size": model.policy.architecture.state_size,
            "feature_dim": model.policy.architecture.output_dim,
            "human_slots": model.policy.architecture.num_entities,
        }
        save_json(run_dir / "resolved_config.json", config)
        print("Stateful architecture loaded:")
        print(f"  robot GRU state: {model.policy.architecture.robot_hidden_size}")
        print(f"  entity GRU state: {model.policy.architecture.entity_hidden_size}")
        print(f"  entity slots: {model.policy.architecture.num_entities}")
        print(f"  packed recurrent state: {model.policy.architecture.state_size}")
        learn_model(config, model, build_callbacks(config, run_dir))
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stateful social-context-fusion with recurrent PPO.")
    parser.add_argument("--config", default="stateful_training_pipeline/config.yaml")
    arguments = parser.parse_args()
    print(f"Training complete. Run data saved to: {train(arguments.config)}")


if __name__ == "__main__":
    main()
