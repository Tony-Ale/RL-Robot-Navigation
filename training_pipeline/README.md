# PPO Training Pipeline

This package contains a config-driven Stable-Baselines3 PPO pipeline for the
SocNavGym hierarchical navigation project.

The intended flow is:

```text
SocNavGym observation dict
    -> selected architecture feature extractor
    -> PPO actor/critic heads
    -> [linear_velocity, angular_velocity]
    -> DifferentialDriveActionWrapper
    -> SocNavGym [linear, 0, angular]
```

The project reward is configured through SocNavGym's native `reward_file`
setting in the scenario YAML. The default training scenario points to
`custom_rewards/socnavgym_social_safety_reward.py`, whose `Reward(RewardAPI)`
class returns the social-safety and checkpoint reward directly.

## Run

This project currently uses a pinned CPU-friendly dependency stack:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.0+cpu torchvision==0.19.0+cpu
python -m pip install --no-deps stable-baselines3==2.4.1 gymnasium==1.0.0 numpy==1.26.4 typing-extensions==4.15.0 tensorboard
```

This keeps Stable-Baselines3 available while preserving compatibility with DGL
and the existing SocNavGym stack.

Stable-Baselines3 uses Gymnasium. The default config enables a small
compatibility adapter for SocNavGym:

```yaml
environment:
    gymnasium_compatibility: true
```

Then start training:

```bash
.venv/bin/python -m training_pipeline.train --config training_pipeline/config.yaml
```

Every run creates its own folder under `runs/`:

```text
runs/<timestamp>_<experiment_name>/
    checkpoints/
    metrics/
    testing/
    tensorboard/
    experiment_config.yaml
    resolved_config.json
    metadata.json
    training_time.json
    training_seed_history.json
```

To continue a run across days, set:

```yaml
experiment:
    resume_run_dir: "runs/<existing_run_folder>"
training:
    resume_from_checkpoint: "runs/<existing_run_folder>/checkpoints/ppo_step_100000.zip"
    reset_num_timesteps: false
```

When `resume_run_dir` is set, new checkpoints and metrics are appended inside
that same run folder. `training_time.json` tracks a running wall-clock session,
marks it completed after the final checkpoint save, and updates cumulative
training time. With `reset_num_timesteps: false`, progress/ETA uses a global
target of `resume_start_step + training.total_timesteps`; for example, resuming
from step `100352` for another `100000` steps targets about `200352`.
Each launch also appends the active `experiment.seed` and resume/checkpoint
fields to `training_seed_history.json`; choose resumed training seeds manually
by editing `experiment.seed`.

## Metrics

PPO agent metrics are handled by Stable-Baselines3 TensorBoard logging.

Navigation training metrics are collected from SocNavGym's `info` dict at the
end of each episode and are written to both:

```text
TensorBoard: navigation/train/*
CSV: runs/<run>/metrics/navigation_training_metrics.csv
```

Navigation evaluation metrics are collected during periodic deterministic
evaluation episodes and saved to:

```text
CSV: runs/<run>/metrics/navigation_evaluation_metrics.csv
TensorBoard: navigation/eval/*_mean
```

By default, evaluation reuses a fixed validation seed set at every evaluation
interval. The first seed is `evaluation.eval_seed_base`, and episode seeds then
increment by one. The evaluation CSV records the seed used for each episode.

## Final Testing

The separate `testing_pipeline/` package can test a trained learned agent on
held-out seeds and compare it with the SocNavGym ORCA robot baseline.

Run testing for an existing run with:

```bash
.venv/bin/python -m testing_pipeline.runner --config training_pipeline/config.yaml --run-dir runs/<run>
```

To run testing automatically after training, enable:

```yaml
testing:
    enabled: true
    run_after_training: true
    fixed_test_seeds: true
    compare_with_baseline: true
    baseline: "orca"
```

Testing writes:

```text
runs/<run>/testing/test_agent_metrics.csv
runs/<run>/testing/test_orca_metrics.csv
runs/<run>/testing/agent_vs_orca_comparison.csv
runs/<run>/testing/test_summary.json
```

The comparison is learned agent vs ORCA robot baseline. PPO is only the training
algorithm used to produce the learned agent. Repeated final test runs overwrite
the configured testing CSV/JSON files by default to avoid duplicate rows. If a
fixed test seed cannot produce waypoint guidance, testing skips that seed and
continues to later seeds until it has the requested number of valid episodes;
ORCA comparison uses the learned agent's valid seed list.

## Offline Checkpoint Evaluation

For cleaner learning curves, evaluate saved checkpoints after training with a
fixed set of validation seeds:

```bash
.venv/bin/python -m testing_pipeline.evaluate_checkpoints \
    --config testing_pipeline/offline_evaluation_config.yaml
```

The offline evaluator reads its own config, rebuilds the environment from the
training config, resolves checkpoints from the selected run folder, and writes:

```text
runs/<run>/offline_evaluation/checkpoint_episode_metrics.csv
runs/<run>/offline_evaluation/checkpoint_summary_metrics.csv
runs/<run>/offline_evaluation/checkpoint_orca_comparison.csv
runs/<run>/offline_evaluation/plots/
```

`checkpoint_source` can select all checkpoints, only `ppo_step_*.zip`, only
`ppo_final_step_*.zip`, or a named list. With `overwrite_existing: false`, the
evaluator skips checkpoints already present in the summary CSV, so resumed
training can add new checkpoints without repeating older evaluations.

## Render A Trained Agent

To watch a learned agent act in the SocNavGym renderer:

```bash
.venv/bin/python -m testing_pipeline.render_agent --config testing_pipeline/render_config.yaml
```

Set `rendering.run_dir` in `testing_pipeline/render_config.yaml`. The script
loads the latest `ppo_final_step_*.zip` checkpoint by default. All runtime
settings, including checkpoint, seed, episode count, deterministic actions,
delay, policy type, and overlays, are configured in that file. CLI values remain
available as optional overrides:

```bash
.venv/bin/python -m testing_pipeline.render_agent \
    --config testing_pipeline/render_config.yaml \
    --run-dir runs/<run> \
    --checkpoint runs/<run>/checkpoints/ppo_final_step_100352.zip \
    --seed 10042 \
    --episodes 1
```

Set `rendering.warning_zones` or `rendering.path_waypoints` to enable the
corresponding overlays. Their CLI flags provide temporary overrides:

```bash
.venv/bin/python -m testing_pipeline.render_agent \
    --config testing_pipeline/render_config.yaml \
    --run-dir runs/<run> \
    --seed 10042 \
    --episodes 1 \
    --path-waypoints
```

When rendering with fixed seeds and waypoint features, seeds that cannot produce
waypoint guidance are skipped so `--episodes` still means valid rendered
episodes.

## Architecture Selection

Choose the architecture in YAML:

```yaml
architecture:
    name: "social_context_fusion"
    config_path: "architectures/social_context_fusion/config.yaml"
```

Available names:

```text
social_context_fusion
feedforward_social_context_fusion
joint_pair_context_fusion
hybrid_context_fusion
dual_context_fusion
joint_scene_fusion
crowd_context_fusion
```

The architecture is used as PPO's feature extractor. PPO still learns the final
actor head that outputs differential-drive actions.

`feedforward_social_context_fusion` preserves the masking, interaction,
attention, reduction, and fusion stages of `social_context_fusion`, while MLPs
encode only the current robot and entity observations.

## Entity History And Walls

`observation_history.temporal_entity_keys` selects entity keys whose stable
slots retain genuine recent observations. Keys omitted from that list repeat
their current rows across time. All-wall observations may use genuine history;
nearest-wall observations may not because distance ranking can change slot
identity between steps.

The wall wrapper supports two modes:

```yaml
nearest_wall_segments:
    enabled: true
    mode: "all"       # "nearest" or "all"
    count: 24         # Fixed capacity; all mode never silently truncates.
    include_boundary_walls: true  # False keeps corridor walls only.
    observation_key: "walls"
```

`nearest` keeps the closest segments by surface clearance. `all` preserves
SocNavGym wall and segment order, pads below capacity, and raises an error above
capacity. Boundary filtering identifies perimeter walls from room geometry and
supports square and rectangular rooms. The wall wrapper owns wall extraction and padding; the fixed-space
wrapper continues to handle SocNavGym's normally padded entity keys.
