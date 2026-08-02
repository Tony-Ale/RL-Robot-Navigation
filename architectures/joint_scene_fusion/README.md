# Joint Scene Fusion Network

This folder contains a second social-navigation architecture for comparison with `architectures/social_context_fusion`.

The key difference is that this model uses one shared BiGRU. For each entity, the model concatenates the robot and entity observations from the same timestep:

```text
[robot_observation, entity_observation]
```

That pair sequence is passed through one BiGRU to produce a scene/entity context.

## Inputs

```python
robot_observation: [batch, robot_time, robot_input_dim]
entity_history:    [batch, entities, entity_time, entity_input_dim]
entity_mask:       [batch, entities]
```

For convenience, these shapes are also accepted:

```python
robot_observation: [batch, robot_input_dim]
entity_history:    [batch, entities, entity_input_dim]
```

Robot and entity histories must have equal lengths so the BiGRU receives aligned
`[robot_t, entity_t]` pairs. The separate robot projection uses the latest robot frame.

## Architecture

Current configured dimensions are:

```text
robot observation: 9-D
entity observation: 14-D
robot/entity pair input to BiGRU: 23-D
single BiGRU hidden per direction: 32
scene/entity context h_scene: 64-D
robot projection: 64-D
[robot_context, scene_context]: 128-D
interaction embedding u_k: 100-D
reduced feature lambda_k: 64-D
final [robot_context, social_context]: 128-D
final head hidden sizes: [150, 100, 100]
```

The attention and feature-reduction path mirrors `SocialContextFusionNet`, so experiments can compare whether separate robot/entity BiGRUs or one joint pair BiGRU works better.

## Example

```python
import torch
from architectures.joint_scene_fusion import JointSceneFusionNet

model = JointSceneFusionNet.from_yaml("architectures/joint_scene_fusion/config.yaml")

robot = torch.randn(4, 9)
entities = torch.randn(4, 6, 1, 14)
mask = torch.ones(4, 6, dtype=torch.bool)

result = model(robot, entities, mask)
features = torch.cat([result["robot_context"], result["social_context"]], dim=-1)
attention = result["attention_weights"]
```
