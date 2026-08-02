import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

from training_pipeline.utils import configure_matplotlib_cache, load_yaml

configure_matplotlib_cache()

from testing_pipeline.runner import (
    is_planner_reset_failure,
    load_learned_agent_policy,
    reset_env,
    reset_policy,
    resolve_checkpoint_path,
    test_seeds,
)
from training_pipeline.env_factory import make_eval_env


def prepare_render_config(
    config: Dict[str, Any],
    enable_warning_zones: bool = False,
    enable_path_waypoints: bool = False,
) -> Dict[str, Any]:
    """Return a copy of config with optional render-only visualization enabled."""
    prepared = _deep_copy(config)
    wrappers = prepared.setdefault("wrappers", {})
    if enable_path_waypoints:
        astar_cfg = wrappers.setdefault("astar", {})
        astar_cfg["enabled"] = True
        nav_cfg = wrappers.setdefault("navigation_features", {})
        nav_cfg.setdefault("config", {}).setdefault("visualization", {})["enabled"] = True
    if enable_warning_zones:
        wrappers.setdefault("warning_zone_visualization", {})["enabled"] = True
    return prepared


def render_agent(
    config: Dict[str, Any],
    run_dir: Path,
    checkpoint_path: Optional[Path] = None,
    seed: Optional[int] = None,
    episodes: int = 1,
    deterministic: bool = True,
    delay_seconds: float = 0.0,
    enable_warning_zones: bool = False,
    enable_path_waypoints: bool = False,
    policy_type: Optional[str] = None,
    device: str = "auto",
    video_path: Optional[Path] = None,
    video_fps: float = 10.0,
) -> None:
    """Load a learned agent, render its behavior, and optionally save those frames."""
    render_config = prepare_render_config(
        config,
        enable_warning_zones=enable_warning_zones,
        enable_path_waypoints=enable_path_waypoints,
    )
    checkpoint = resolve_checkpoint_path(render_config.get("testing", {}), run_dir, checkpoint_path)
    seeds = _render_seeds(render_config, seed=seed, episodes=episodes)

    recorder = None if video_path is None else RenderedVideoRecorder(video_path, video_fps)
    env = make_eval_env(render_config)
    try:
        policy_settings = {
            "deterministic": deterministic,
            "device": device,
        }
        if policy_type is not None:
            policy_settings["policy_type"] = policy_type
        policy = load_learned_agent_policy(checkpoint, env, policy_settings, render_config)
        success_count = 0
        episode_index = 0
        seed_index = 0
        next_seed = None if not seeds or seeds[-1] is None else int(seeds[-1]) + 1
        while episode_index < episodes:
            if seed_index < len(seeds):
                episode_seed = seeds[seed_index]
                seed_index += 1
            elif next_seed is not None:
                episode_seed = next_seed
                next_seed += 1
            else:
                break

            try:
                obs, _ = reset_env(env, episode_seed)
            except RuntimeError as exc:
                if episode_seed is not None and is_planner_reset_failure(exc):
                    print(f"Skipping render seed {episode_seed}: no waypoints generated.")
                    continue
                raise

            reset_policy(policy)
            episode_index += 1
            done = False
            episode_reward = 0.0
            episode_length = 0
            final_info = {}
            print(f"Rendering episode {episode_index}/{episodes} seed={episode_seed}")

            while not done:
                action = policy.predict(obs, env=env)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                if recorder is not None:
                    recorder.write(_current_render_frame(env))
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                episode_length += 1
                final_info = info
                if delay_seconds > 0:
                    time.sleep(delay_seconds)

            if _info_flag(final_info, "SUCCESS"):
                success_count += 1
            print(
                "Episode finished: "
                f"reward={episode_reward:.3f}, "
                f"length={episode_length}, "
                f"success={final_info.get('SUCCESS')}, "
                f"collision={final_info.get('COLLISION')}, "
                f"timeout={final_info.get('TIMEOUT')}"
            )

        total_episodes = episode_index
        success_rate = success_count / total_episodes if total_episodes else 0.0
        print(
            "Render summary: "
            f"successes={success_count}/{total_episodes}, "
            f"success_rate={success_rate:.1%}"
        )
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def render_from_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> None:
    """Render from a dedicated YAML file, with optional CLI-provided overrides."""
    wrapper_config = load_yaml(config_path)
    settings = dict(wrapper_config.get("rendering", wrapper_config.get("render", {})))
    if not settings:
        raise ValueError("Render config must contain a 'rendering' section.")
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})

    training_config_path = settings.get("training_config_path")
    if not training_config_path:
        raise ValueError("rendering.training_config_path is required.")
    run_dir = settings.get("run_dir")
    if not run_dir:
        raise ValueError("rendering.run_dir is required.")

    training_config = load_yaml(training_config_path)
    checkpoint = settings.get("checkpoint_path", settings.get("checkpoint"))
    record_video = bool(settings.get("record_video", False))
    configured_video_path = settings.get("video_path")
    if record_video and not configured_video_path:
        raise ValueError("rendering.video_path is required when rendering.record_video is true.")
    render_agent(
        config=training_config,
        run_dir=Path(run_dir),
        checkpoint_path=None if not checkpoint else Path(checkpoint),
        seed=settings.get("seed"),
        episodes=int(settings.get("episodes", 1)),
        deterministic=bool(settings.get("deterministic", True)),
        delay_seconds=float(settings.get("delay_seconds", 0.0)),
        enable_warning_zones=bool(settings.get("warning_zones", False)),
        enable_path_waypoints=bool(settings.get("path_waypoints", False)),
        policy_type=settings.get("policy_type"),
        device=str(settings.get("device", "auto")),
        video_path=Path(configured_video_path) if record_video else None,
        video_fps=float(settings.get("video_fps", 10.0)),
    )


class RenderedVideoRecorder:
    """Write the completed SocNavGym render frames to one MP4 file."""

    def __init__(self, path: Path, fps: float):
        self.path = Path(path)
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("rendering.video_fps must be greater than zero.")
        if self.path.suffix.lower() != ".mp4":
            raise ValueError("rendering.video_path must use the .mp4 extension.")
        self.writer = None
        self.frame_size = None
        self.frame_count = 0

    def write(self, frame) -> None:
        if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
            raise RuntimeError("SocNavGym did not produce a three-channel rendered frame for video recording.")

        height, width = frame.shape[:2]
        frame_size = (int(width), int(height))
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.frame_size = frame_size
            codec = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), codec, self.fps, frame_size)
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise RuntimeError(f"OpenCV could not open the MP4 video writer: {self.path}")
        elif frame_size != self.frame_size:
            raise RuntimeError(
                f"Rendered frame size changed from {self.frame_size} to {frame_size} while recording."
            )

        self.writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        print(f"Video saved: {self.path} ({self.frame_count} frames at {self.fps:g} FPS)")


def _current_render_frame(env):
    """Return the frame after SocNavGym and all render callbacks have drawn it."""
    base_env = getattr(env, "unwrapped", env)
    frame = getattr(base_env, "world_image", None)
    if frame is None:
        raise RuntimeError(
            "Video recording requires env.render() to populate env.unwrapped.world_image."
        )
    return frame.copy()


def _render_seeds(config: Dict[str, Any], seed: Optional[int], episodes: int):
    if seed is not None:
        return [int(seed) + index for index in range(int(episodes))]
    return test_seeds(config, {**config.get("testing", {}), "n_test_episodes": int(episodes)})


def _deep_copy(value):
    if isinstance(value, dict):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deep_copy(item) for item in value]
    return value


def _info_flag(info: Dict[str, Any], key: str) -> bool:
    value = info.get(key, False)
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def main():
    parser = argparse.ArgumentParser(description="Render a trained learned navigation agent in SocNavGym.")
    parser.add_argument("--config", default="testing_pipeline/render_config.yaml", help="Path to render YAML config.")
    parser.add_argument("--run-dir", default=None, help="Optional override for rendering.run_dir.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path. Defaults to latest final checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Optional first episode seed. Defaults to testing seed config.")
    parser.add_argument("--episodes", type=int, default=None, help="Optional episode-count override.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of using deterministic actions.")
    parser.add_argument("--delay-seconds", type=float, default=None, help="Optional delay override.")
    parser.add_argument("--warning-zones", action="store_true", default=None, help="Enable warning-zone render overlay.")
    parser.add_argument("--path-waypoints", action="store_true", default=None, help="Enable A* path and waypoint overlays.")
    args = parser.parse_args()

    render_from_config(
        args.config,
        overrides={
            "run_dir": args.run_dir,
            "checkpoint_path": args.checkpoint,
            "seed": args.seed,
            "episodes": args.episodes,
            "deterministic": False if args.stochastic else None,
            "delay_seconds": args.delay_seconds,
            "warning_zones": args.warning_zones,
            "path_waypoints": args.path_waypoints,
        },
    )


if __name__ == "__main__":
    main()
