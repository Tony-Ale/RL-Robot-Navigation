# Crowd Context Fusion Network

This folder contains another architecture for the training comparison set.

The key difference from `architectures/social_context_fusion` is the robot branch. Instead of
encoding the robot observation with a BiGRU, this model embeds the current robot
observation with an MLP. Human/entity observations are still encoded with one
shared BiGRU crowd encoder.

## Inputs

```python
robot_observation: [batch, robot_input_dim]
entity_history:    [batch, entities, entity_time, entity_input_dim]
entity_mask:       [batch, entities]
```

For convenience, these shapes are also accepted:

```python
robot_observation: [batch, robot_time, robot_input_dim]
entity_history:    [batch, entities, entity_input_dim]
```

If a robot history is provided, the model uses the latest robot observation.

## Architecture

Current configured dimensions are:

```text
robot observation: 9-D
robot MLP hidden sizes: [150, 100]
h_robot: 64-D
entity observation: 14-D
crowd BiGRU hidden per direction: 32
h_crowd: 64-D
[h_robot, h_crowd]: 128-D
interaction embedding u_k: 100-D
reduced feature lambda_k: 64-D
final [h_robot, social_context]: 128-D
final head hidden sizes: [150, 100, 100]
```

The attention and feature-reduction path mirrors `SocialContextFusionNet`, so
you can compare the effect of replacing the robot BiGRU with a robot MLP while
keeping the rest of the model family familiar.

## Example

```python
import torch
from architectures.crowd_context_fusion import CrowdContextFusionNet

model = CrowdContextFusionNet.from_yaml("architectures/crowd_context_fusion/config.yaml")

robot = torch.randn(4, 9)
entities = torch.randn(4, 6, 5, 14)
mask = torch.ones(4, 6, dtype=torch.bool)

result = model(robot, entities, mask)
features = torch.cat([result["robot_context"], result["social_context"]], dim=-1)
attention = result["attention_weights"]
```
