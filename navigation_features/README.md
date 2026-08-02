# Navigation Feature Wrapper

`CoordinateFrameWaypointWrapper` adds planner-derived waypoint features to a SocNavGym observation dictionary.

The intended stack is:

```python
env = gym.make("SocNavGym-v1", config="...")
env = SocNavAStarWrapper(env, config_path="global_planning/astar_wrapper_config.yaml")
env = CoordinateFrameWaypointWrapper(env, config_path="navigation_features/config.yaml")
```

## Coordinate Frames

The wrapper supports two robot-centric frames:

- `heading_aligned`: the x-axis follows the robot heading. This matches the standard SocNavGym robot-centric observation convention, so observations are left unchanged.
- `goal_aligned`: the x-axis points from the robot to its goal. The wrapper rotates robot goal coordinates, entity positions, and entity orientation sine/cosine pairs into this frame.

## Waypoint Features

Waypoint features are generated using a sequential sliding-window strategy: waypoints are kept in path order and truncated to the next `max_waypoints` upcoming points. If fewer points are available, the last available waypoint is repeated until the fixed-size window is full. Reset-time episodes with no planner waypoints are skipped internally up to `max_reset_attempts`, so PPO does not train on planner-failed resets. Explicit seeded resets fail immediately when no waypoints are produced, because retrying the same seed repeats the same scenario. Unexpected planner errors are raised instead of being retried as empty plans. The first observation after reset shows the initial path window; later observations advance the window after `step()` when the robot is within `advance_radius` of the next waypoint, when the robot has clearly passed the active waypoint along the path direction, or when the reward function has already advanced the shared `last_reached_waypoint_index`.

Each waypoint can include:

- x and y position in the active frame
- distance from the robot
- bearing in the active frame
- optional sine and cosine of the bearing

By default, the feature vector is added to the observation dictionary under `waypoint_features`. Re-reading the feature vector does not advance the window. During PPO training, `ArchitectureFeaturesExtractor` can merge that vector into the robot observation before the selected neural architecture processes it.

## Waypoint Window Visualization

The wrapper can draw the current policy-facing waypoint window during render:

```yaml
visualization:
    enabled: true
```

This overlay draws the same fixed-size waypoint slots used to build `waypoint_features`, including repeated final slots when fewer than `max_waypoints` remain. Current window slots are drawn as colored rings, and already reached/passed waypoints can be drawn as muted skipped-waypoint rings. It is different from the A* overlay, which can draw the full sampled path.
