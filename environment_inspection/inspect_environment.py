import argparse
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from testing_pipeline.policies import ORCARobotPolicy
from testing_pipeline.policy_loading import load_learned_agent_policy
from training_pipeline.architecture_extractor import _entity_mask
from training_pipeline.episode_runtime import reset_policy
from training_pipeline.env_factory import make_socnav_env
from training_pipeline.utils import load_yaml
from environment_inspection.failure_analysis import run_failure_analysis
from environment_inspection.stall_analysis import run_stall_analysis
from environment_inspection.trajectory_analysis import run_trajectory_analysis
from environment_inspection.wall_segment_analysis import run_wall_segment_analysis


CONFIG_PATH = Path(__file__).with_name("config.yaml")


def inspect_from_config(config_path=CONFIG_PATH):
    config = load_yaml(config_path)
    pipeline_config = load_yaml(config["pipeline_config"])
    analysis_mode = _analysis_mode(config)

    if analysis_mode == "wall_segment_analysis":
        run_wall_segment_analysis(config, pipeline_config)
        return

    pipeline_config = _prepare_pipeline_config(config, pipeline_config)
    env = make_socnav_env(pipeline_config)
    try:
        latest_obs = {"value": None}
        if config.get("visualization", {}).get("nearest_walls", False):
            _install_nearest_wall_overlay(env, latest_obs, config, pipeline_config)

        policy = _make_policy(config, env)
        if analysis_mode == "trajectory_analysis":
            run_trajectory_analysis(env, policy, config)
            return

        if analysis_mode == "stall_analysis":
            run_stall_analysis(env, policy, config)
            return

        if analysis_mode == "failure_analysis":
            run_failure_analysis(env, policy, config)
            return

        seed = int(config["episode"]["seed"])
        max_steps = int(config["episode"]["max_steps"])
        verbosity = int(config.get("debug", {}).get("verbosity", 1))

        obs, _ = env.reset(seed=seed)
        reset_policy(policy)
        latest_obs["value"] = obs
        totals = _empty_reward_totals()

        print(f"Environment inspection started: policy={config['policy']['type']} seed={seed} max_steps={max_steps}")
        _print_observation_summary(obs, config, verbosity)

        for step in range(1, max_steps + 1):
            action = policy.predict(obs, env) if hasattr(policy, "predict") else env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            latest_obs["value"] = obs
            _update_reward_totals(totals, reward, info)

            print(f"\nstep={step} action={_format_array(action)} reward={float(reward):.6f} terminated={terminated} truncated={truncated}")
            _print_reward_breakdown(info, totals, verbosity)
            _print_observation_summary(obs, config, verbosity)

            if config.get("render", {}).get("enabled", False):
                env.render()
                _wait_after_render(config)

            if terminated or truncated:
                print("\nEpisode ended.")
                break

        print("\nFinal reward totals:")
        for key, value in totals.items():
            print(f"  {key}: {value:.6f}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect a SocNavGym policy and environment episode.")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    args = parser.parse_args()
    inspect_from_config(args.config)


def _analysis_mode(config):
    mode_names = (
        "failure_analysis",
        "stall_analysis",
        "wall_segment_analysis",
        "trajectory_analysis",
    )
    enabled = [name for name in mode_names if config.get(name, {}).get("enabled", False)]
    if len(enabled) > 1:
        raise ValueError(
            "Only one environment inspection analysis mode may be enabled at a time: "
            + ", ".join(enabled)
        )
    return enabled[0] if enabled else None


def _make_policy(config: Dict[str, Any], env):
    policy_type = config["policy"]["type"]
    if policy_type == "random":
        return None
    if policy_type == "orca":
        return ORCARobotPolicy()
    if policy_type in {"ppo", "stateful_ppo"}:
        checkpoint = config["policy"].get("checkpoint")
        if not checkpoint:
            raise ValueError(
                f"environment_inspection policy.checkpoint is required when policy.type is {policy_type}."
            )
        checkpoint_path = _checkpoint_path(checkpoint, policy_type)
        policy = load_learned_agent_policy(
            checkpoint_path,
            env,
            {
                "policy_type": policy_type,
                "deterministic": config["policy"].get("deterministic", True),
                "device": config["policy"].get("device", "auto"),
            },
        )
        _validate_policy_spaces(policy.model, env)
        return policy
    raise ValueError(
        'environment_inspection policy.type must be "random", "orca", "ppo", or "stateful_ppo".'
    )


def _checkpoint_path(checkpoint, policy_type):
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f'environment_inspection policy.type "{policy_type}" checkpoint does not exist: {path}'
        )
    resolved = path.resolve()
    print(f"Loading {policy_type} checkpoint: {resolved}")
    return resolved


def _empty_reward_totals():
    return {
        "total_reward": 0.0,
        "progress_reward": 0.0,
        "checkpoint_reward": 0.0,
        "warning_zone_reward": 0.0,
        "static_warning_zone_reward": 0.0,
        "stagnation_penalty": 0.0,
        "goal_reward": 0.0,
        "timeout_reward": 0.0,
        "collision_reward": 0.0,
        "shaped_reward": 0.0,
    }


def _update_reward_totals(totals, reward, info):
    reward = float(reward)
    reason = str(info.get("reward_reason", "unknown"))
    totals["total_reward"] += reward

    if reason == "shaped":
        warning = float(info.get("warning_zone_reward", 0.0))
        static_warning = float(info.get("static_warning_zone_reward", 0.0))
        checkpoint = float(info.get("checkpoint_reward", 0.0))
        stagnation = float(info.get("stagnation_penalty", 0.0))
        progress = float(
            info.get(
                "distance_reward",
                reward - warning - static_warning - checkpoint - stagnation,
            )
        )
        totals["progress_reward"] += progress
        totals["checkpoint_reward"] += checkpoint
        totals["warning_zone_reward"] += warning
        totals["static_warning_zone_reward"] += static_warning
        totals["stagnation_penalty"] += stagnation
        totals["shaped_reward"] += reward
    elif reason == "goal":
        totals["goal_reward"] += reward
    elif reason == "timeout":
        totals["timeout_reward"] += reward
    elif "collision" in reason:
        totals["collision_reward"] += reward


def _print_reward_breakdown(info, totals, verbosity):
    if verbosity < 1:
        return
    reason = info.get("reward_reason", "unknown")
    print(f"  reward_reason={reason}")
    if verbosity >= 2:
        _print_info_group(
            "components",
            info,
            (
                "custom_reward",
                "distance_reward",
                "checkpoint_reward",
                "warning_zone_reward",
                "static_warning_zone_reward",
                "stagnation_penalty",
            ),
        )
        _print_info_group(
            "safety",
            info,
            (
                "warning_zone_hits",
                "static_warning_zone_hits",
                "nearest_static_clearance",
                "nearest_static_type",
                "stagnation_stalled",
                "stagnation_displacement",
            ),
        )
        _print_info_group(
            "navigation",
            info,
            (
                "checkpoint_hits",
                "goal_distance",
                "progress_target",
                "progress_target_index",
                "progress_target_distance",
            ),
        )
        print(
            "  running_totals: "
            + ", ".join(
                f"{key}={value:.4f}"
                for key, value in totals.items()
                if abs(value) > 1e-12 or key == "total_reward"
            )
        )


def _print_info_group(label, info, keys):
    values = [
        f"{key}={_format_value(info[key])}"
        for key in keys
        if key in info and info[key] is not None
    ]
    if values:
        print(f"  {label}: " + ", ".join(values))


def _prepare_pipeline_config(config, pipeline_config):
    """Apply inspection-only visualization choices without mutating training config."""
    visualization = config.get("visualization", {})
    if "warning_zones" not in visualization:
        return pipeline_config

    prepared = deepcopy(pipeline_config)
    warning_wrapper = prepared.setdefault("wrappers", {}).setdefault(
        "warning_zone_visualization",
        {},
    )
    warning_wrapper["enabled"] = bool(visualization["warning_zones"])
    return prepared


def _print_observation_summary(obs, config, verbosity):
    if verbosity < 1 or not isinstance(obs, dict):
        return

    entity_keys = config.get("debug", {}).get("entity_keys", [])
    entity_feature_dim = int(config.get("debug", {}).get("entity_feature_dim", 14))
    shapes = {key: tuple(np.asarray(value).shape) for key, value in obs.items()}
    print("  observations: " + ", ".join(f"{key}={shape}" for key, shape in shapes.items()))

    if verbosity < 2:
        return

    present = []
    missing = []
    for key in entity_keys:
        if key not in obs:
            missing.append(key)
            continue
        rows, history_length = _entity_view(obs[key], entity_feature_dim)
        mask = _mask_rows(rows)
        real_count = int(mask.sum())
        details = f"real={real_count}, padded={len(mask) - real_count}, capacity={len(mask)}"
        if history_length > 1:
            details += f", history={history_length}"
        present.append(f"    {key}: {details}")
        if verbosity >= 3:
            present.append(f"      mask={mask.astype(int).tolist()}")
            present.append(f"      latest_rows=\n{rows}")

    if present:
        print("  entities:")
        print("\n".join(present))
    if missing:
        print(f"  missing entity keys: {', '.join(missing)}")

    if "waypoint_features" in obs and verbosity >= 2:
        if verbosity >= 3:
            print(f"  waypoint_features=\n{_format_array(obs['waypoint_features'])}")


def _entity_rows(values, entity_feature_dim):
    rows, _ = _entity_view(values, entity_feature_dim)
    return rows


def _entity_view(values, entity_feature_dim):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, entity_feature_dim), dtype=np.float32), 1
    if values.ndim == 3:
        if values.shape[-1] != entity_feature_dim:
            raise ValueError(
                f"Temporal entity observation must end with feature dimension {entity_feature_dim}, "
                f"got {values.shape}."
            )
        return values[:, -1, :], values.shape[1]
    if values.ndim == 2 and values.shape[-1] == entity_feature_dim:
        return values, 1

    values = values.reshape(-1)
    if values.size == 0:
        return np.zeros((0, entity_feature_dim), dtype=np.float32)
    if values.size % entity_feature_dim != 0:
        raise ValueError(f"Entity observation size {values.size} is not divisible by {entity_feature_dim}.")
    return values.reshape(-1, entity_feature_dim), 1


def _mask_rows(rows):
    if rows.size == 0:
        return np.zeros((0,), dtype=bool)
    tensor = torch.as_tensor(rows, dtype=torch.float32).unsqueeze(0)
    return _entity_mask(tensor, mask_zero_entities=True).squeeze(0).cpu().numpy()


def _validate_policy_spaces(model, env):
    from stable_baselines3.common.utils import check_for_correct_spaces

    try:
        check_for_correct_spaces(env, model.observation_space, model.action_space)
    except ValueError as exc:
        raise ValueError(
            "The checkpoint observation/action spaces do not match the inspection environment. "
            "Set pipeline_config in the selected inspection YAML to the saved training config "
            "for this checkpoint. Checkpoints trained without observation history cannot inspect "
            "a history-enabled environment."
        ) from exc


def _install_nearest_wall_overlay(env, latest_obs, config, pipeline_config):
    base_env = env.unwrapped
    callbacks = getattr(base_env, "render_callbacks", None)
    if callbacks is None:
        callbacks = []
        setattr(base_env, "render_callbacks", callbacks)

    viz_cfg = config.get("visualization", {})
    color = tuple(int(v) for v in viz_cfg.get("nearest_wall_color_bgr", [255, 255, 0]))
    radius = int(viz_cfg.get("nearest_wall_radius", 5))
    thickness = int(viz_cfg.get("nearest_wall_thickness", 2))
    frame_mode = _navigation_frame_mode(pipeline_config)

    def draw_nearest_walls(image, render_env):
        obs = latest_obs.get("value")
        if not isinstance(obs, dict) or "walls" not in obs:
            return
        rows = _entity_rows(obs["walls"], 14)
        rows = rows[_mask_rows(rows)]
        if rows.size == 0:
            return

        import cv2
        from socnavgym.envs.utils.utils import w2px, w2py

        angle = _active_frame_angle(render_env, frame_mode)
        c = math.cos(angle)
        s = math.sin(angle)
        for row in rows:
            x_frame, y_frame = float(row[6]), float(row[7])
            x_world = render_env.robot.x + c * x_frame - s * y_frame
            y_world = render_env.robot.y + s * x_frame + c * y_frame
            center = (
                w2px(x_world, render_env.PIXEL_TO_WORLD_X, render_env.MAP_X),
                w2py(y_world, render_env.PIXEL_TO_WORLD_Y, render_env.MAP_Y),
            )
            cv2.circle(image, center, radius, color, thickness)

    callbacks.append(draw_nearest_walls)


def _active_frame_angle(env, frame_mode):
    if frame_mode == "goal_aligned":
        return math.atan2(env.robot.goal_y - env.robot.y, env.robot.goal_x - env.robot.x)
    return float(getattr(env.robot, "orientation", 0.0))


def _navigation_frame_mode(pipeline_config):
    nav_cfg = pipeline_config.get("wrappers", {}).get("navigation_features", {})
    inline_config = nav_cfg.get("config") or {}
    if inline_config.get("coordinate_frame", {}).get("mode"):
        return inline_config["coordinate_frame"]["mode"]
    config_path = nav_cfg.get("config_path")
    if config_path:
        return load_yaml(config_path).get("coordinate_frame", {}).get("mode")
    return None


def _wait_after_render(config):
    wait_ms = int(config.get("render", {}).get("wait_ms", 1))
    if wait_ms <= 0:
        return
    try:
        import cv2

        cv2.waitKey(wait_ms)
    except Exception:
        pass


def _format_array(value):
    return np.array2string(np.asarray(value), precision=3, suppress_small=True)


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


if __name__ == "__main__":
    main()
