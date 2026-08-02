# Environment Inspection

This folder contains config-driven tools for checking how the wrapped SocNavGym environment behaves before or after training.

Run the inspector with:

```bash
.venv/bin/python -m environment_inspection.inspect_environment
```

All options are controlled from `environment_inspection/config.yaml`.

## Inspection Modes

Only one analysis mode should normally be enabled at a time.

- `failure_analysis.enabled`: runs many seeded episodes and summarizes the robot's initial clearance from the nearest entity for failed episodes.
- `stall_analysis.enabled`: runs many seeded episodes and summarizes rolling robot displacement to help tune the stagnation penalty.
- `wall_segment_analysis.enabled`: compares calculated wall capacity with live SocNavGym wall rows over a deterministic seed sweep.
- `trajectory_analysis.enabled`: replays explicit seeds and saves top-down trajectory snapshots.
- If all analysis modes are disabled, the tool runs one rendered step-by-step inspection episode.

When `wall_segment_analysis.enabled` is true, wall analysis is the selected mode;
the inspector exits after the deterministic wall sweep without loading a policy
or running reward diagnostics.

## Reward Diagnostics

The step-by-step inspector separates reward output into three compact groups:

- `components`: progress, checkpoint, human warning-zone, static warning-zone, and stagnation contributions.
- `safety`: human/static warning hits, nearest static surface clearance and type, and stagnation state.
- `navigation`: goal distance and active waypoint progress information.

Running totals keep `warning_zone_reward` and
`static_warning_zone_reward` separate. Static warning penalties are also removed
from the fallback progress calculation, so they are never reported as navigation
progress.

Set `visualization.warning_zones: true` to enable the pipeline's existing
warning-zone render wrapper only for inspection. Human sectors and enabled
static-object footprints are then drawn from the same reward configuration used
by the environment. `visualization.nearest_walls` separately marks the centres
of the wall rows currently exposed in `obs["walls"]`; it works with both
`nearest` and `all` wall modes and uses the latest frame when history is enabled.

## Wall Segment Analysis

Wall segment analysis does not load a policy. For every configured segment size,
it calculates the maximum capacity from the room geometry and checks live wall
observations over the configured seed range. The summary reports the calculated
capacity, largest observed row count, and the first seed that produced it.
`include_boundary_walls` selects whether both perimeter and corridor walls are
counted or only corridor walls are counted.

The calculation supports square and rectangular rooms. All-mode wall capacity
must be at least the reported calculated value; the wall wrapper raises an error
instead of silently discarding rows when capacity is too small.

## Trajectory Snapshots

Trajectory analysis uses the same unified policy loader as normal inspection, so
`policy.type` may be `ppo`, `stateful_ppo`, `orca`, or `random`. Stateful policy
memory is reset before every selected episode.

Each seed produces `trajectory.csv`, `summary.json`, and
`trajectory_snapshot.png`. The snapshot overlays the robot's recorded
world-coordinate path and the reset-time A* path on the physical scene at the
start of the episode. Dynamic entities are therefore shown only at their initial
positions; the figure does not imply that they remained there. The final marker
states the observed terminal outcome. When `require_astar` is true, a missing
A* wrapper or plan raises an error instead of silently drawing an unavailable
reference.

## Stall Analysis Output

The stall analysis records the robot position after reset and after every step, then checks how far the robot moved over each configured `window_steps` value.

### Displacement Summary

Example:

```text
window | failed_min_m median/p25 | success_min_m median/p25
    12 |           0.0340/0.0120 |            0.2200/0.1500
```

- `window`: the number of environment steps used for the rolling displacement check.
- `failed_min_m`: minimum rolling displacement per failed episode, summarized across failed episodes.
- `success_min_m`: minimum rolling displacement per successful episode, summarized across successful episodes.
- `median`: the middle value across episodes.
- `p25`: the 25th percentile across episodes. Lower values mean stronger low-movement behaviour.

### Threshold Hit Summary

Example:

```text
window threshold | failed_hit_rate | success_hit_rate | failed_longest median/max
    12    0.1000 |   18/25    72.00% |    3/25    12.00% |    4.0/18
```

- `threshold`: candidate `min_displacement` value in meters.
- `failed_hit_rate`: how many failed episodes had at least one rolling window with displacement below `threshold`.
- `success_hit_rate`: how many successful episodes also had at least one rolling window below `threshold`.
- `failed_longest median/max`: for failed episodes, the median and maximum number of consecutive stalled windows.

### Most Common Failed Stall Range

Example:

```text
window_steps=12 min_displacement=0.1000m caught_failed=18/25 (72.00%) caught_success=3/25 (12.00%)
```

- `window_steps`: candidate stagnation window size.
- `min_displacement`: candidate stagnation displacement threshold.
- `caught_failed`: failed episodes where this setting detected at least one stall.
- `caught_success`: successful episodes where this setting also detected at least one stall.

A useful setting has high `caught_failed` and low `caught_success`. If `caught_success` is high, the penalty may punish normal careful behaviour such as yielding, rotating, or slowing near humans.
