from typing import Any, Dict, List

import numpy as np

from training_pipeline.callbacks import NavigationEvaluationCallback
from training_pipeline.episode_runtime import is_planner_reset_failure
from training_pipeline.metrics import NAVIGATION_METRIC_KEYS


class RecurrentNavigationEvaluationCallback(NavigationEvaluationCallback):
    """Navigation evaluation that carries and resets recurrent policy state."""

    def _run_evaluation(self) -> None:
        self.evaluation_count += 1
        rows: List[Dict[str, Any]] = []
        episode = 1
        seed_attempt = 0
        maximum_attempts = max(self.n_eval_episodes * 10, self.n_eval_episodes)

        while episode <= self.n_eval_episodes:
            if seed_attempt >= maximum_attempts:
                raise RuntimeError(
                    f"Evaluation could not collect {self.n_eval_episodes} valid episodes "
                    f"after {maximum_attempts} reset attempts."
                )
            seed_attempt += 1
            seed = self._episode_seed(seed_attempt)
            try:
                obs, _ = self.eval_env.reset() if seed is None else self.eval_env.reset(seed=seed)
            except RuntimeError as exc:
                if seed is not None and is_planner_reset_failure(exc):
                    continue
                raise

            state = None
            episode_start = np.ones((1,), dtype=bool)
            done = False
            episode_reward = 0.0
            episode_length = 0
            final_info: Dict[str, Any] = {}
            while not done:
                action, state = self.model.predict(
                    obs,
                    state=state,
                    episode_start=episode_start,
                    deterministic=self.deterministic,
                )
                episode_start[:] = False
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
            rows.append(row)
            episode += 1

        self._log_evaluation_summary(rows)
