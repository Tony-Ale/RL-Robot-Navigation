import math
from dataclasses import dataclass, field

from custom_rewards.static_obstacle_warning_zone import (
    StaticObstacleWarningZoneConfig,
    compute_static_obstacle_warning_zone,
    load_static_obstacle_warning_zone_config,
)

from navigation_features.waypoint_state import (
    CURRENT_WAYPOINTS_ATTR,
    LAST_REACHED_WAYPOINT_ATTR,
    PROGRESS_TARGET_DISTANCE_ATTR,
    PROGRESS_TARGET_SIGNATURE_ATTR,
    WAYPOINT_ADVANCE_RADIUS_ATTR,
    WAYPOINT_SIGNATURE_ATTR,
    advance_waypoint_index,
)


STAGNATION_POSITION_HISTORY_ATTR = "_social_safety_stagnation_position_history"
STAGNATION_LAST_TICK_ATTR = "_social_safety_stagnation_last_tick"


@dataclass
class SocialSafetyRewardConfig:
    goal_reward: float = 10.0
    timeout_penalty: float = -10.0
    collision_penalty: float = -0.1
    human_collision_penalty: float = -1.0
    object_collision_penalty: float = -0.1
    goal_progress_scale: float = 0.01
    checkpoint_reward: float = 0.5
    waypoint_advance_radius: float = 0.3
    checkpoint_reward_enabled: bool = True
    warning_zone_scale: float = 0.1
    radius_speed_scale: float = 0.8
    gait_width_scale: float = 0.5
    angle_scale_pi: float = 2.2
    angle_speed_decay: float = 1.7
    angle_offset_pi: float = 0.2
    heading_angle_offset: float = 0.0
    default_human_radius: float = 0.3
    include_static_humans: bool = True
    include_dynamic_humans: bool = True
    include_interaction_humans: bool = True
    static_obstacle_warning_zone: StaticObstacleWarningZoneConfig = field(
        default_factory=StaticObstacleWarningZoneConfig
    )
    stagnation_penalty_enabled: bool = False
    stagnation_window_steps: int = 12
    stagnation_min_displacement: float = 0.1
    stagnation_penalty: float = -0.02


def load_social_safety_reward_config(config_path):
    """Load reward parameters from a commented YAML config file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load custom reward YAML config files.") from exc

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    reward_values = data.get("reward", {})
    checkpoint_values = data.get("checkpoint_reward", {})
    warning_zone_values = data.get("dynamic_warning_zone", {})
    human_values = data.get("humans", {})
    static_warning_values = data.get("static_obstacle_warning_zone", {})
    stagnation_values = data.get("stagnation_penalty", {})

    return SocialSafetyRewardConfig(
        goal_reward=reward_values.get("goal_reward", SocialSafetyRewardConfig.goal_reward),
        timeout_penalty=reward_values.get("timeout_penalty", SocialSafetyRewardConfig.timeout_penalty),
        collision_penalty=reward_values.get("collision_penalty", SocialSafetyRewardConfig.collision_penalty),
        human_collision_penalty=reward_values.get(
            "human_collision_penalty",
            SocialSafetyRewardConfig.human_collision_penalty,
        ),
        object_collision_penalty=reward_values.get(
            "object_collision_penalty",
            reward_values.get("collision_penalty", SocialSafetyRewardConfig.object_collision_penalty),
        ),
        goal_progress_scale=reward_values.get("goal_progress_scale", SocialSafetyRewardConfig.goal_progress_scale),
        checkpoint_reward=checkpoint_values.get("reward", SocialSafetyRewardConfig.checkpoint_reward),
        waypoint_advance_radius=checkpoint_values.get(
            "advance_radius",
            checkpoint_values.get("radius", SocialSafetyRewardConfig.waypoint_advance_radius),
        ),
        checkpoint_reward_enabled=checkpoint_values.get("enabled", SocialSafetyRewardConfig.checkpoint_reward_enabled),
        warning_zone_scale=warning_zone_values.get("warning_zone_scale", SocialSafetyRewardConfig.warning_zone_scale),
        radius_speed_scale=warning_zone_values.get("radius_speed_scale", SocialSafetyRewardConfig.radius_speed_scale),
        gait_width_scale=warning_zone_values.get("gait_width_scale", SocialSafetyRewardConfig.gait_width_scale),
        angle_scale_pi=warning_zone_values.get("angle_scale_pi", SocialSafetyRewardConfig.angle_scale_pi),
        angle_speed_decay=warning_zone_values.get("angle_speed_decay", SocialSafetyRewardConfig.angle_speed_decay),
        angle_offset_pi=warning_zone_values.get("angle_offset_pi", SocialSafetyRewardConfig.angle_offset_pi),
        heading_angle_offset=warning_zone_values.get("heading_angle_offset", SocialSafetyRewardConfig.heading_angle_offset),
        default_human_radius=human_values.get("default_human_radius", SocialSafetyRewardConfig.default_human_radius),
        include_static_humans=human_values.get("include_static_humans", SocialSafetyRewardConfig.include_static_humans),
        include_dynamic_humans=human_values.get("include_dynamic_humans", SocialSafetyRewardConfig.include_dynamic_humans),
        include_interaction_humans=human_values.get("include_interaction_humans", SocialSafetyRewardConfig.include_interaction_humans),
        static_obstacle_warning_zone=load_static_obstacle_warning_zone_config(static_warning_values),
        stagnation_penalty_enabled=stagnation_values.get(
            "enabled",
            SocialSafetyRewardConfig.stagnation_penalty_enabled,
        ),
        stagnation_window_steps=stagnation_values.get(
            "window_steps",
            SocialSafetyRewardConfig.stagnation_window_steps,
        ),
        stagnation_min_displacement=stagnation_values.get(
            "min_displacement",
            SocialSafetyRewardConfig.stagnation_min_displacement,
        ),
        stagnation_penalty=stagnation_values.get("penalty", SocialSafetyRewardConfig.stagnation_penalty),
    )


def compute_social_safety_reward(
    env,
    previous_goal_distance,
    config=None,
    reached_goal=None,
    timeout=None,
    collision=None,
):
    """
    Compute the dynamic-warning-zone reward.

    Args:
        env: SocNavGym-like environment with robot and human state.
        previous_goal_distance: Robot-goal distance at the previous timestep, used only when waypoint progress is unavailable.
        config: Optional SocialSafetyRewardConfig.
        reached_goal: Optional precomputed terminal flag.
        timeout: Optional precomputed timeout flag.
        collision: Optional precomputed collision flag.

    Returns:
        tuple[float, dict]: reward and diagnostic components.
    """
    cfg = config or SocialSafetyRewardConfig()

    current_goal_distance = _distance(env.robot.x, env.robot.y, env.robot.goal_x, env.robot.goal_y)

    if reached_goal is None:
        goal_threshold = getattr(env, "GOAL_THRESHOLD", getattr(env, "GOAL_RADIUS", 0.0))
        reached_goal = current_goal_distance < goal_threshold
    if timeout is None:
        timeout = getattr(env, "ticks", 0) > getattr(env, "EPISODE_LENGTH", math.inf)
    if collision is None:
        collision = _robot_has_collision(env)

    if reached_goal:
        return cfg.goal_reward, _info("goal", cfg.goal_reward, current_goal_distance, 0.0)
    if timeout:
        return cfg.timeout_penalty, _info("timeout", cfg.timeout_penalty, current_goal_distance, 0.0)
    if collision:
        reason, penalty = _collision_reward(env, cfg)
        return penalty, _info(reason, penalty, current_goal_distance, 0.0)

    waypoints = _current_waypoints(env)
    reached_waypoint_hits = _advance_active_waypoint_if_reached(env, waypoints, cfg)
    checkpoint_reward, checkpoint_hits = _checkpoint_reward_contribution(reached_waypoint_hits, cfg)
    progress_reward, progress_info = _progress_reward_contribution(
        env,
        waypoints,
        cfg,
        previous_goal_distance,
        current_goal_distance,
    )

    warning_reward = 0.0
    warning_hits = 0
    for human in _iter_humans(env, cfg):
        contribution = _warning_zone_contribution(env.robot, human, cfg)
        if contribution is not None:
            warning_reward += contribution
            warning_hits += 1

    static_warning_reward, static_warning_info = compute_static_obstacle_warning_zone(
        env, cfg.static_obstacle_warning_zone
    )
    stagnation_reward, stagnation_info = _stagnation_penalty_contribution(env, cfg)
    shaped_reward = progress_reward + checkpoint_reward + warning_reward + static_warning_reward + stagnation_reward
    return shaped_reward, _info(
        "shaped",
        shaped_reward,
        current_goal_distance,
        warning_reward,
        warning_hits,
        checkpoint_reward,
        checkpoint_hits,
        progress_target=progress_info["target"],
        progress_target_index=progress_info["target_index"],
        progress_target_distance=progress_info["target_distance"],
        stagnation_penalty=stagnation_reward,
        stagnation_stalled=stagnation_info["stalled"],
        stagnation_displacement=stagnation_info["displacement"],
        **static_warning_info,
    )


def _stagnation_penalty_contribution(env, cfg):
    if not cfg.stagnation_penalty_enabled:
        return 0.0, {"stalled": False, "displacement": None}

    window_steps = int(cfg.stagnation_window_steps)
    if window_steps <= 0:
        return 0.0, {"stalled": False, "displacement": None}

    history = _stagnation_position_history(env)
    current_position = (float(env.robot.x), float(env.robot.y))
    history.append(current_position)
    if len(history) > window_steps + 1:
        del history[: len(history) - (window_steps + 1)]

    if len(history) <= window_steps:
        return 0.0, {"stalled": False, "displacement": None}

    previous_position = history[0]
    displacement = _distance(current_position[0], current_position[1], previous_position[0], previous_position[1])
    stalled = displacement < float(cfg.stagnation_min_displacement)
    return (float(cfg.stagnation_penalty) if stalled else 0.0), {
        "stalled": stalled,
        "displacement": displacement,
    }


def reset_stagnation_tracking(env):
    robot = getattr(env, "robot", None)
    if robot is None:
        setattr(env, STAGNATION_POSITION_HISTORY_ATTR, [])
    else:
        setattr(env, STAGNATION_POSITION_HISTORY_ATTR, [(float(robot.x), float(robot.y))])
    setattr(env, STAGNATION_LAST_TICK_ATTR, getattr(env, "ticks", 0))


def _stagnation_position_history(env):
    tick = getattr(env, "ticks", None)
    last_tick = getattr(env, STAGNATION_LAST_TICK_ATTR, None)
    if tick is not None:
        setattr(env, STAGNATION_LAST_TICK_ATTR, tick)

    history = getattr(env, STAGNATION_POSITION_HISTORY_ATTR, None)
    if history is None or (tick is not None and last_tick is not None and tick <= last_tick):
        history = []
        setattr(env, STAGNATION_POSITION_HISTORY_ATTR, history)
    return history


def _checkpoint_reward_contribution(hits, cfg):
    if not cfg.checkpoint_reward_enabled:
        return 0.0, 0
    return hits * cfg.checkpoint_reward, hits


def _advance_active_waypoint_if_reached(env, waypoints, cfg):
    if not waypoints:
        return 0

    _sync_waypoint_tracking(env, waypoints)
    radius = _waypoint_advance_radius(env, cfg)
    if radius <= 0:
        return 0
    # hits can be greater than 1 in edge cases such as waypoints are close together, or robot moves a large distance in one env step etc.
    last_reached, hits = advance_waypoint_index(
        env.robot.x,
        env.robot.y,
        waypoints,
        getattr(env, LAST_REACHED_WAYPOINT_ATTR, -1),
        radius,
    )

    setattr(env, LAST_REACHED_WAYPOINT_ATTR, last_reached)
    return hits


def _sync_waypoint_tracking(env, waypoints):
    signature = _waypoint_signature(waypoints)
    previous_signature = getattr(env, WAYPOINT_SIGNATURE_ATTR, None)
    if previous_signature == signature:
        return

    setattr(env, WAYPOINT_SIGNATURE_ATTR, signature)
    setattr(env, LAST_REACHED_WAYPOINT_ATTR, -1)
    _clear_progress_target_state(env)


def _waypoint_advance_radius(env, cfg):
    radius = getattr(env, WAYPOINT_ADVANCE_RADIUS_ATTR, None)
    if radius is None:
        radius = cfg.waypoint_advance_radius
    return float(radius or 0.0)


def _clear_progress_target_state(env):
    if hasattr(env, PROGRESS_TARGET_SIGNATURE_ATTR):
        delattr(env, PROGRESS_TARGET_SIGNATURE_ATTR)
    if hasattr(env, PROGRESS_TARGET_DISTANCE_ATTR):
        delattr(env, PROGRESS_TARGET_DISTANCE_ATTR)


def _progress_reward_contribution(env, waypoints, cfg, previous_goal_distance, current_goal_distance):
    """Reward progress toward the first unreached waypoint, falling back to the global goal."""
    target = _active_progress_target(env, waypoints)
    target_signature = _progress_target_signature(target)
    current_distance = _distance(env.robot.x, env.robot.y, target["x"], target["y"])

    previous_signature = getattr(env, PROGRESS_TARGET_SIGNATURE_ATTR, None)
    previous_distance = getattr(env, PROGRESS_TARGET_DISTANCE_ATTR, None)

    if previous_signature == target_signature and previous_distance is not None:
        reward = cfg.goal_progress_scale * (previous_distance - current_distance)
    elif target["kind"] == "goal":
        reward = cfg.goal_progress_scale * (previous_goal_distance - current_goal_distance)
    else:
        reward = 0.0

    setattr(env, PROGRESS_TARGET_SIGNATURE_ATTR, target_signature)
    setattr(env, PROGRESS_TARGET_DISTANCE_ATTR, current_distance)

    return reward, {
        "target": target["kind"],
        "target_index": target["index"],
        "target_distance": current_distance,
    }


def _active_progress_target(env, waypoints):
    if waypoints:
        _sync_waypoint_tracking(env, waypoints)
    last_reached = int(getattr(env, LAST_REACHED_WAYPOINT_ATTR, -1))

    target_index = last_reached + 1
    if target_index < len(waypoints):
        waypoint = waypoints[target_index]
        return {
            "kind": "waypoint",
            "index": target_index,
            "x": waypoint[0],
            "y": waypoint[1],
        }

    return {
        "kind": "goal",
        "index": None,
        "x": float(env.robot.goal_x),
        "y": float(env.robot.goal_y),
    }


def _progress_target_signature(target):
    return (
        target["kind"],
        target["index"],
        round(target["x"], 4),
        round(target["y"], 4),
    )


def _current_waypoints(env):
    waypoints = getattr(env, CURRENT_WAYPOINTS_ATTR, None)
    if waypoints is not None:
        return [(float(point[0]), float(point[1])) for point in waypoints]

    if hasattr(env, "get_waypoints"):
        waypoints = env.get_waypoints()
    else:
        plan = getattr(env, "latest_plan", None)
        waypoints = getattr(plan, "waypoints", None)
        if waypoints is None:
            waypoints = getattr(plan, "path_world", None)
    if waypoints is None:
        return []
    return [(float(point[0]), float(point[1])) for point in waypoints]


def _waypoint_signature(waypoints):
    return tuple((round(x, 4), round(y, 4)) for x, y in waypoints)


def _warning_zone_contribution(robot, human, cfg):
    activation_radius = _warning_zone_activation_radius(human, robot, cfg)
    distance_to_human = _distance(robot.x, robot.y, human.x, human.y)

    if distance_to_human > activation_radius:
        return None
    if not _robot_inside_human_warning_sector(robot, human, cfg):
        return None

    exponent = distance_to_human - activation_radius
    return cfg.warning_zone_scale * (math.exp(exponent) - 1.0)


def _dynamic_warning_radius(human, cfg):
    speed = abs(float(getattr(human, "speed", 0.0) or 0.0))
    step_length = _human_step_length(human, cfg)
    return cfg.radius_speed_scale * speed + step_length


def _warning_zone_activation_radius(human, robot, cfg):
    return max(0.0, _dynamic_warning_radius(human, cfg) + _robot_radius(robot))


def _human_step_length(human, cfg):
    width = getattr(human, "width", None)
    if width is None:
        return cfg.default_human_radius
    return cfg.gait_width_scale * float(width)


def _dynamic_warning_angle(human, cfg):
    speed = abs(float(getattr(human, "speed", 0.0) or 0.0))
    return cfg.angle_scale_pi * math.pi * math.exp(-cfg.angle_speed_decay * speed) + cfg.angle_offset_pi * math.pi


def _robot_inside_human_warning_sector(robot, human, cfg):
    sector_angle = _dynamic_warning_angle(human, cfg)
    if sector_angle >= 2.0 * math.pi:
        return True

    angle_to_robot = math.atan2(robot.y - human.y, robot.x - human.x)
    human_heading = _warning_zone_heading(human, cfg)
    return abs(_angle_difference(angle_to_robot, human_heading)) <= sector_angle / 2.0


def _warning_zone_heading(human, cfg):
    heading = float(getattr(human, "orientation", 0.0) or 0.0)
    return heading + cfg.heading_angle_offset


def _robot_radius(robot):
    radius = getattr(robot, "radius", 0.0)
    return float(radius or 0.0)


def _iter_humans(env, cfg):
    if cfg.include_static_humans:
        yield from getattr(env, "static_humans", [])
    if cfg.include_dynamic_humans:
        yield from getattr(env, "dynamic_humans", [])
    if cfg.include_interaction_humans:
        for interaction in getattr(env, "static_interactions", []) + getattr(env, "moving_interactions", []):
            yield from getattr(interaction, "humans", [])
        for interaction in getattr(env, "h_l_interactions", []):
            human = getattr(interaction, "human", None)
            if human is not None:
                yield human


def _robot_has_collision(env):
    return _collision_kind(env) is not None


def _collision_reward(env, cfg):
    kind = _collision_kind(env)
    if kind == "human":
        return "human_collision", cfg.human_collision_penalty
    if kind == "object":
        return "object_collision", cfg.object_collision_penalty
    return "collision", cfg.object_collision_penalty


def _collision_kind(env):
    robot = env.robot
    humans = getattr(env, "static_humans", []) + getattr(env, "dynamic_humans", [])
    if any(robot.collides(human) for human in humans):
        return "human"

    interactions = getattr(env, "moving_interactions", []) + getattr(env, "static_interactions", []) + getattr(env, "h_l_interactions", [])
    for interaction in interactions:
        if _interaction_human_collision(robot, interaction):
            return "human"

    objects = getattr(env, "plants", []) + getattr(env, "walls", []) + getattr(env, "tables", []) + getattr(env, "laptops", [])
    if any(robot.collides(obj) for obj in objects):
        return "object"

    if any(interaction.collides(robot) for interaction in interactions):
        return "object"

    return None


def _interaction_human_collision(robot, interaction):
    human = getattr(interaction, "human", None)
    if human is not None and robot.collides(human):
        return True

    return any(robot.collides(human) for human in getattr(interaction, "humans", []))


def _distance(x1, y1, x2, y2):
    return math.hypot(x1 - x2, y1 - y2)


def _angle_difference(angle_a, angle_b):
    return math.atan2(math.sin(angle_a - angle_b), math.cos(angle_a - angle_b))


def _info(
    reason,
    reward,
    goal_distance,
    warning_reward,
    warning_hits=0,
    checkpoint_reward=0.0,
    checkpoint_hits=0,
    progress_target=None,
    progress_target_index=None,
    progress_target_distance=None,
    stagnation_penalty=0.0,
    stagnation_stalled=False,
    stagnation_displacement=None,
    static_warning_zone_reward=0.0,
    static_warning_zone_hits=0,
    nearest_static_clearance=None,
    nearest_static_type=None,
):
    return {
        "reward_reason": reason,
        "reward": reward,
        "goal_distance": goal_distance,
        "warning_zone_reward": warning_reward,
        "warning_zone_hits": warning_hits,
        "checkpoint_reward": checkpoint_reward,
        "checkpoint_hits": checkpoint_hits,
        "progress_target": progress_target,
        "progress_target_index": progress_target_index,
        "progress_target_distance": progress_target_distance,
        "stagnation_penalty": stagnation_penalty,
        "stagnation_stalled": stagnation_stalled,
        "stagnation_displacement": stagnation_displacement,
        "static_warning_zone_reward": static_warning_zone_reward,
        "static_warning_zone_hits": static_warning_zone_hits,
        "nearest_static_clearance": nearest_static_clearance,
        "nearest_static_type": nearest_static_type,
    }
