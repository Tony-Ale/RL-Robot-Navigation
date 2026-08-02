"""Authoritative learned-policy loading for testing and inspection."""

from pathlib import Path
from typing import Any, Dict, Optional

from testing_pipeline.policies import LearnedAgentPolicy


SUPPORTED_POLICY_TYPES = {"ppo", "stateful_ppo"}


def resolve_policy_type(settings: Dict[str, Any], training_config: Optional[Dict[str, Any]] = None) -> str:
    """Resolve the checkpoint implementation from an explicit setting or architecture metadata."""
    configured = settings.get("policy_type")
    if configured is not None:
        policy_type = str(configured).strip().lower()
    else:
        architecture_name = str((training_config or {}).get("architecture", {}).get("name", ""))
        policy_type = "stateful_ppo" if architecture_name == "stateful_social_context_fusion" else "ppo"

    if policy_type not in SUPPORTED_POLICY_TYPES:
        supported = ", ".join(sorted(SUPPORTED_POLICY_TYPES))
        raise ValueError(f"policy_type must be one of: {supported}.")
    return policy_type


def load_learned_agent_policy(
    checkpoint_path: Path,
    env,
    settings: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None,
):
    """Load ordinary or stateful PPO behind the common evaluation policy interface."""
    deterministic = bool(settings.get("deterministic", True))
    device = settings.get("device", "auto")
    policy_type = resolve_policy_type(settings, training_config)

    if policy_type == "stateful_ppo":
        from stateful_training_pipeline.policies import load_stateful_policy

        return load_stateful_policy(
            Path(checkpoint_path),
            env=env,
            deterministic=deterministic,
            device=device,
        )

    from stable_baselines3 import PPO

    model = PPO.load(str(checkpoint_path), env=env, device=device)
    model.policy.set_training_mode(False)
    return LearnedAgentPolicy(model, deterministic=deterministic)
