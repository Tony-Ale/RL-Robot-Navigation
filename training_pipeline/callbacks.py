import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from training_pipeline.utils import configure_matplotlib_cache

configure_matplotlib_cache()

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from training_pipeline.episode_runtime import is_planner_reset_failure
from training_pipeline.metrics import CSVMetricWriter, NAVIGATION_METRIC_KEYS, count_existing_rows


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_training_time(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {"total_wall_clock_seconds": 0.0, "sessions": []}
    with open(path, "r") as f:
        data = json.load(f)
    data.setdefault("total_wall_clock_seconds", 0.0)
    data.setdefault("sessions", [])
    return data


def _save_training_time(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)


def _safe_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(number) or np.isinf(number):
        return 0.0
    return number


def _format_optional_number(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def record_training_time_session(
    run_dir: Path,
    started_at: str,
    ended_at: str,
    wall_clock_seconds: float,
    start_timesteps: int,
    end_timesteps: int,
    session_id: Optional[str] = None,
    status: str = "completed",
) -> Dict[str, Any]:
    """Append a wall-clock training session and update cumulative time."""
    path = Path(run_dir) / "training_time.json"
    data = _load_training_time(path)
    session = {
        "session_id": session_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_clock_seconds": float(wall_clock_seconds),
        "start_timesteps": int(start_timesteps),
        "end_timesteps": int(end_timesteps),
    }
    previous_seconds = 0.0
    updated_existing = False
    if session_id is not None:
        for index, existing in enumerate(data["sessions"]):
            if existing.get("session_id") == session_id:
                previous_seconds = float(existing.get("wall_clock_seconds", 0.0))
                data["sessions"][index] = session
                updated_existing = True
                break
    if not updated_existing:
        data["sessions"].append(session)
    data["total_wall_clock_seconds"] = (
        float(data.get("total_wall_clock_seconds", 0.0))
        - previous_seconds
        + float(wall_clock_seconds)
    )
    _save_training_time(path, data)
    return data


class NavigationTrainingCallback(BaseCallback):
    """Log ETA, checkpoints, and per-episode navigation metrics."""

    def __init__(
        self,
        run_dir: Path,
        total_timesteps: int,
        checkpoint_interval_steps: int,
        eta_log_interval_steps: int,
        navigation_csv_name: str,
        reset_num_timesteps: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.run_dir = Path(run_dir)
        self.requested_timesteps = int(total_timesteps)
        self.total_timesteps = int(total_timesteps)
        self.reset_num_timesteps = bool(reset_num_timesteps)
        self.checkpoint_interval_steps = int(checkpoint_interval_steps)
        self.eta_log_interval_steps = int(eta_log_interval_steps)
        self.last_checkpoint_step = 0
        self.last_eta_step = 0
        self.start_time = None
        self.session_started_at = None
        self.session_id = None
        self.session_start_timesteps = 0
        fieldnames = ["episode", "global_step", "episode_reward", "episode_length"] + NAVIGATION_METRIC_KEYS
        csv_path = self.run_dir / "metrics" / navigation_csv_name
        self.episode_count = count_existing_rows(csv_path)
        self.last_episode_reward = None
        self.last_episode_steps = None
        self.csv_writer = CSVMetricWriter(csv_path, fieldnames)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.session_started_at = _utc_now_iso()
        self.session_id = f"{self.session_started_at}_step_{int(self.num_timesteps)}"
        self.session_start_timesteps = int(self.num_timesteps)
        if self.reset_num_timesteps:
            self.total_timesteps = self.requested_timesteps
        else:
            self.total_timesteps = self.session_start_timesteps + self.requested_timesteps
        self.last_checkpoint_step = self.session_start_timesteps
        self.last_eta_step = self.session_start_timesteps
        self._record_training_time_snapshot(status="running")

    def _on_step(self) -> bool:
        self._record_finished_episodes()
        self._maybe_log_eta()
        self._maybe_save_checkpoint()
        return True

    def _maybe_log_eta(self) -> None:
        if self.num_timesteps - self.last_eta_step < self.eta_log_interval_steps:
            return
        self.last_eta_step = self.num_timesteps
        elapsed = max(time.time() - (self.start_time or time.time()), 1e-8)
        session_steps = max(self.num_timesteps - self.session_start_timesteps, 0)
        steps_per_second = session_steps / elapsed
        remaining = max(self.total_timesteps - self.num_timesteps, 0)
        eta_seconds = remaining / max(steps_per_second, 1e-8)
        self.logger.record("time/steps_per_second", steps_per_second)
        self.logger.record("time/eta_seconds", eta_seconds)
        if self.last_episode_reward is not None:
            self.logger.record("navigation/train/last_episode_reward", self.last_episode_reward)
        if self.last_episode_steps is not None:
            self.logger.record("navigation/train/last_episode_steps", self.last_episode_steps)
        self._record_training_time_snapshot(status="running")
        if self.verbose:
            print(
                f"Training progress: {self.num_timesteps}/{self.total_timesteps} steps, "
                f"episode reward {_format_optional_number(self.last_episode_reward)}, "
                f"episode steps {self.last_episode_steps if self.last_episode_steps is not None else 'n/a'}, "
                f"{steps_per_second:.2f} steps/s, "
                f"ETA {eta_seconds / 60:.1f} min"
            )

    def _maybe_save_checkpoint(self) -> None:
        if self.checkpoint_interval_steps <= 0:
            return
        if self.num_timesteps - self.last_checkpoint_step < self.checkpoint_interval_steps:
            return
        self.last_checkpoint_step = self.num_timesteps
        path = self.run_dir / "checkpoints" / f"ppo_step_{self.num_timesteps}.zip"
        self.model.save(path)

    def _record_finished_episodes(self) -> None:
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])
        for env_index, done in enumerate(dones):
            if not done:
                continue
            info = infos[env_index] if env_index < len(infos) else {}
            self.episode_count += 1
            monitor_episode = info.get("episode", {})
            episode_reward = monitor_episode.get("r", rewards[env_index] if env_index < len(rewards) else None)
            episode_length = monitor_episode.get("l", None)
            row = {
                "episode": self.episode_count,
                "global_step": self.num_timesteps,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
            }
            self.last_episode_reward = _safe_float(episode_reward)
            self.last_episode_steps = int(_safe_float(episode_length)) if episode_length not in (None, "") else None
            for key in NAVIGATION_METRIC_KEYS:
                row[key] = info.get(key)
                if isinstance(info.get(key), (int, float, bool, np.integer, np.floating, np.bool_)):
                    self.logger.record(f"navigation/train/{key}", float(info[key]))
            self.csv_writer.write(row)
            self.logger.record("navigation/train/episode", self.episode_count)
            self.logger.record("navigation/train/last_episode_reward", self.last_episode_reward)
            if self.last_episode_steps is not None:
                self.logger.record("navigation/train/last_episode_steps", self.last_episode_steps)

    def _record_training_time_snapshot(self, status: str) -> Dict[str, Any]:
        end_time = time.time()
        started_at = self.session_started_at or _utc_now_iso()
        ended_at = _utc_now_iso()
        wall_clock_seconds = max(end_time - (self.start_time or end_time), 0.0)
        return record_training_time_session(
            self.run_dir,
            started_at=started_at,
            ended_at=ended_at,
            wall_clock_seconds=wall_clock_seconds,
            start_timesteps=self.session_start_timesteps,
            end_timesteps=int(self.num_timesteps),
            session_id=self.session_id,
            status=status,
        )

    def _on_training_end(self) -> None:
        final_path = self.run_dir / "checkpoints" / f"ppo_final_step_{self.num_timesteps}.zip"
        self.model.save(final_path)
        timing = self._record_training_time_snapshot(status="completed")
        wall_clock_seconds = timing["sessions"][-1]["wall_clock_seconds"]
        self.logger.record("time/session_wall_clock_seconds", wall_clock_seconds)
        self.logger.record("time/total_wall_clock_seconds", timing["total_wall_clock_seconds"])


class TrainingRenderCallback(BaseCallback):
    """Render the training environment at a fixed step interval."""

    def __init__(self, render_interval_steps: int = 1):
        super().__init__()
        self.render_interval_steps = max(1, int(render_interval_steps))
        self.last_render_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_render_step >= self.render_interval_steps:
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method("render")
            else:
                self.training_env.render()
            self.last_render_step = self.num_timesteps
        return True


class NavigationEvaluationCallback(BaseCallback):
    """Run deterministic evaluation episodes and save navigation metrics to CSV."""

    def __init__(
        self,
        eval_env,
        run_dir: Path,
        eval_interval_steps: int,
        n_eval_episodes: int,
        deterministic: bool,
        navigation_csv_name: str,
        fixed_episode_seeds: bool = True,
        eval_seed_base: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.run_dir = Path(run_dir)
        self.eval_interval_steps = int(eval_interval_steps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.deterministic = bool(deterministic)
        self.fixed_episode_seeds = bool(fixed_episode_seeds)
        self.eval_seed_base = None if eval_seed_base is None else int(eval_seed_base)
        self.last_eval_step = 0
        self.evaluation_count = 0
        fieldnames = ["evaluation", "global_step", "episode", "seed", "episode_reward", "episode_length"] + NAVIGATION_METRIC_KEYS
        self.csv_writer = CSVMetricWriter(self.run_dir / "metrics" / navigation_csv_name, fieldnames)

    def _on_training_start(self) -> None:
        self.last_eval_step = int(self.num_timesteps)

    def _on_step(self) -> bool:
        if self.eval_interval_steps <= 0:
            return True
        if self.num_timesteps - self.last_eval_step < self.eval_interval_steps:
            return True
        self.last_eval_step = self.num_timesteps
        self._run_evaluation()
        return True

    def _run_evaluation(self) -> None:
        self.evaluation_count += 1
        episode_rows = []
        episode = 1
        seed_attempt = 0
        max_seed_attempts = max(self.n_eval_episodes * 10, self.n_eval_episodes)
        while episode <= self.n_eval_episodes:
            if seed_attempt >= max_seed_attempts:
                raise RuntimeError(
                    "Evaluation could not collect "
                    f"{self.n_eval_episodes} valid episodes after {max_seed_attempts} reset attempts."
                )
            seed_attempt += 1
            seed = self._episode_seed(seed_attempt)
            try:
                if seed is None:
                    obs, _ = self.eval_env.reset()
                else:
                    obs, _ = self.eval_env.reset(seed=seed)
            except RuntimeError as exc:
                if seed is not None and is_planner_reset_failure(exc):
                    continue
                raise

            done = False
            episode_reward = 0.0
            episode_length = 0
            final_info: Dict[str, Any] = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = bool(terminated or truncated)
                episode_reward += float(reward)
                episode_length += 1
                final_info = info

            row = {
                "evaluation": self.evaluation_count,
                "global_step": self.num_timesteps,
                "episode": episode,
                "seed": seed,
                "episode_reward": episode_reward,
                "episode_length": episode_length,
            }
            for key in NAVIGATION_METRIC_KEYS:
                row[key] = final_info.get(key)
            self.csv_writer.write(row)
            episode_rows.append(row)
            episode += 1

        self._log_evaluation_summary(episode_rows)

    def _log_evaluation_summary(self, rows: List[Dict[str, Any]]) -> None:
        numeric_keys = ["episode_reward", "episode_length"] + NAVIGATION_METRIC_KEYS
        means = {}
        for key in numeric_keys:
            values = [row.get(key) for row in rows]
            values = [float(v) for v in values if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))]
            if values:
                means[key] = float(np.mean(values))
                self.logger.record(f"navigation/eval/{key}_mean", means[key])
        print(
            f"Evaluation @ {self.num_timesteps} steps: "
            f"success {means.get('SUCCESS', 0.0):.2%}, "
            f"collision {means.get('COLLISION', 0.0):.2%}, "
            f"human collision {means.get('COLLISION_HUMAN', 0.0):.2%}, "
            f"object collision {means.get('COLLISION_OBJECT', 0.0):.2%}"
        )

    def _episode_seed(self, episode: int) -> Optional[int]:
        if not self.fixed_episode_seeds or self.eval_seed_base is None:
            return None
        return self.eval_seed_base + int(episode) - 1

    def _on_training_end(self) -> None:
        self.eval_env.close()
