# Stateful Social Context Training

This pipeline trains the standalone `stateful_social_context_fusion` architecture with recurrent PPO. It preserves the original social-context operations while replacing fixed-window BiGRUs with stateful unidirectional robot and per-human GRUs.

The environment must provide padded human observations and disable interaction formation/dispersal so each human slot retains its identity. The observation-history wrapper must remain disabled because recurrent hidden state carries temporal information.

Install the recurrent dependency in the project virtual environment:

```bash
.venv/bin/pip install -r stateful_training_pipeline/requirements.txt
```

Run training:

```bash
.venv/bin/python -m stateful_training_pipeline.train --config stateful_training_pipeline/config.yaml
```

Testing and offline evaluation use the shared `testing_pipeline` entry points.
The archived training config identifies the checkpoint as `stateful_ppo`, so the
shared policy adapter carries recurrent state within an episode and resets it
after every environment reset. Environment inspection uses the unified general
tool with `policy.type: stateful_ppo`:

```bash
.venv/bin/python -m testing_pipeline.runner --config stateful_training_pipeline/config.yaml --run-dir runs/<run>
.venv/bin/python -m testing_pipeline.evaluate_checkpoints --config testing_pipeline/offline_evaluation_config.yaml
.venv/bin/python -m testing_pipeline.render_agent --config testing_pipeline/render_config.yaml
.venv/bin/python -m environment_inspection.inspect_environment --config environment_inspection/config.yaml
```

The existing `training_pipeline` and its standard PPO architectures are unchanged.
