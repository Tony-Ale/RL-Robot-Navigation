# Social Context Fusion Network

This folder contains a PyTorch implementation of a recurrent attention model for social navigation. The model is inspired by the supplied architecture diagram, but is named generically for this dissertation codebase.

## Inputs

The network expects:

```python
robot_history:  [batch, robot_time, robot_input_dim]
entity_history: [batch, entities, entity_time, entity_input_dim]
entity_mask:    [batch, entities]
```

For single-timestep inputs, these shorter shapes are also accepted:

```python
robot_history:  [batch, robot_input_dim]
entity_history: [batch, entities, entity_input_dim]
```

`entity_mask` should be `True` for valid entities and `False` for padded entities.

## Architecture

The model uses:

- a robot BiGRU encoder,
- an entity BiGRU encoder shared across entities,
- an interaction MLP that combines robot and entity recurrent features,
- global mean pooling over entity embeddings,
- an attention MLP,
- a feature-reduction MLP,
- an attention-weighted social context vector,
- a final prediction/value head.

Current configured dimensions are:

```text
BiGRU hidden per direction: 32
robot context: 64-D
entity context: 64-D
joint robot/entity context: 128-D
interaction embedding u_k: 100-D
reduced feature lambda_k: 64-D
final [robot_context, social_context]: 128-D
final head hidden sizes: [150, 100, 100]
```

## Example

```python
import torch
from architectures.social_context_fusion import SocialContextFusionNet

model = SocialContextFusionNet.from_yaml("architectures/social_context_fusion/config.yaml")

robot = torch.randn(4, 1, 9)
entities = torch.randn(4, 6, 1, 14)
mask = torch.ones(4, 6, dtype=torch.bool)

result = model(robot, entities, mask)
features = torch.cat([result["robot_context"], result["social_context"]], dim=-1)
attention = result["attention_weights"]
```

With `prediction_head.enabled: false`, `result["output"]` is `None` and `forward(..., return_attention=False)` returns the fused feature tensor.
