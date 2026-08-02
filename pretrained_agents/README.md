# Pretrained Agents

This folder contains four selected navigation agents and the configuration
snapshots needed to render them from the repository root.

| Folder | Architecture | Environment | Held-out success |
| --- | --- | --- | ---: |
| `persistent_state_humans` | Persistent-state GRU | Humans | 82.2% |
| `persistent_state_humans_plants` | Persistent-state GRU | Humans and plants | 64.0% |
| `feedforward_humans` | Feedforward MLP | Humans | 81.4% |
| `fixed_window_bigru_humans` | Fixed-window BiGRU | Humans | 68.6% |

Run an agent from the repository root:

```bash
.venv/bin/python -m testing_pipeline.render_agent \\
    --config pretrained_agents/persistent_state_humans/render_config.yaml
```

Replace the folder name to render another agent. Each `render_config.yaml`
selects a deterministic successful seed with a comparatively long A* reference
path. CLI arguments can override its seed, episode count, checkpoint, or render
overlays.

To save the rendered episode as an MP4, set `record_video: true` in that
agent's `render_config.yaml`. The configured `video_path` and `video_fps`
control the destination and playback rate.

Each model folder contains:

- `model.zip`: selected Stable-Baselines3 or stateful PPO checkpoint.
- `render_config.yaml`: ready-to-run renderer settings.
- `training_config.yaml`: render-safe pipeline configuration.
- `environment_config.yaml`: packaged SocNavGym scenario.
- `architecture_config.yaml`: packaged network dimensions.
- `reward_config.yaml`: current reward configuration retained for reference.
- `metadata.yaml`: model identity, checksum, provenance, selection evidence, and
  held-out metrics.

Checkpoints use Python/PyTorch serialization. Load checkpoints only from a
trusted source. Verify a model before use with:

```bash
sha256sum pretrained_agents/persistent_state_humans/model.zip
```
