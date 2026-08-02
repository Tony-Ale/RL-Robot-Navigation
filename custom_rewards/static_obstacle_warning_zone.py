import math
from dataclasses import dataclass

from navigation_features.wall_geometry import is_boundary_wall


@dataclass(frozen=True)
class StaticObstacleWarningZoneConfig:
    enabled: bool = False
    warning_clearance: float = 0.25
    warning_zone_scale: float = 0.2
    include_walls: bool = True
    include_boundary_walls: bool = False
    include_tables: bool = False
    include_plants: bool = False
    include_laptops: bool = False


def load_static_obstacle_warning_zone_config(values=None):
    """Build and validate static-obstacle warning settings from YAML values."""
    values = values or {}
    config = StaticObstacleWarningZoneConfig(
        enabled=values.get("enabled", StaticObstacleWarningZoneConfig.enabled),
        warning_clearance=values.get(
            "warning_clearance",
            StaticObstacleWarningZoneConfig.warning_clearance,
        ),
        warning_zone_scale=values.get(
            "warning_zone_scale",
            StaticObstacleWarningZoneConfig.warning_zone_scale,
        ),
        include_walls=values.get("include_walls", StaticObstacleWarningZoneConfig.include_walls),
        include_boundary_walls=values.get(
            "include_boundary_walls",
            StaticObstacleWarningZoneConfig.include_boundary_walls,
        ),
        include_tables=values.get("include_tables", StaticObstacleWarningZoneConfig.include_tables),
        include_plants=values.get("include_plants", StaticObstacleWarningZoneConfig.include_plants),
        include_laptops=values.get("include_laptops", StaticObstacleWarningZoneConfig.include_laptops),
    )
    if config.warning_clearance <= 0.0:
        raise ValueError("static_obstacle_warning_zone.warning_clearance must be greater than zero.")
    if config.warning_zone_scale < 0.0:
        raise ValueError("static_obstacle_warning_zone.warning_zone_scale cannot be negative.")
    return config


def compute_static_obstacle_warning_zone(env, config):
    """Return the nearest static-obstacle penalty and clearance diagnostics."""
    info = {
        "static_warning_zone_reward": 0.0,
        "static_warning_zone_hits": 0,
        "nearest_static_clearance": None,
        "nearest_static_type": None,
    }
    if not config.enabled:
        return 0.0, info

    robot = env.robot
    clearances = [
        (_surface_clearance(robot, obstacle), obstacle)
        for obstacle in _iter_static_obstacles(env, config)
    ]
    if not clearances:
        return 0.0, info

    clearance, nearest = min(clearances, key=lambda item: item[0])
    hits = sum(value <= config.warning_clearance for value, _ in clearances)
    info["static_warning_zone_hits"] = hits
    info["nearest_static_clearance"] = clearance
    info["nearest_static_type"] = getattr(nearest, "name", None)
    if clearance > config.warning_clearance:
        return 0.0, info

    # Collision is handled by the terminal reward branch. Clamping here keeps
    # this shaped contribution bounded at its contact value.
    exponent = max(clearance, 0.0) - config.warning_clearance
    reward = config.warning_zone_scale * (math.exp(exponent) - 1.0)
    info["static_warning_zone_reward"] = reward
    return reward, info


def _iter_static_obstacles(env, config):
    if config.include_walls:
        walls = getattr(env, "walls", [])
        if not config.include_boundary_walls and walls:
            shape = getattr(env, "shape", None)
            if shape not in ("square", "rectangle"):
                raise ValueError("Boundary-wall filtering supports square and rectangle rooms.")
            walls = [
                wall
                for wall in walls
                if not is_boundary_wall(wall, float(env.MAP_X), float(env.MAP_Y))
            ]
        yield from walls
    if config.include_tables:
        yield from getattr(env, "tables", [])
    if config.include_plants:
        yield from getattr(env, "plants", [])
    if config.include_laptops:
        yield from getattr(env, "laptops", [])


def _surface_clearance(robot, obstacle):
    """Return signed robot-body clearance from a supported obstacle surface."""
    robot_radius = float(robot.radius)
    name = getattr(obstacle, "name", None)
    if name == "plant":
        centre_distance = math.hypot(float(robot.x) - float(obstacle.x), float(robot.y) - float(obstacle.y))
        return centre_distance - robot_radius - float(obstacle.radius)
    if name in ("wall", "table", "laptop"):
        width = float(obstacle.thickness if name == "wall" else obstacle.width)
        return _rectangle_clearance(robot, obstacle, width) - robot_radius
    raise ValueError(f"Unsupported static obstacle type: {name!r}.")


def _rectangle_clearance(robot, obstacle, width):
    dx = float(robot.x) - float(obstacle.x)
    dy = float(robot.y) - float(obstacle.y)
    orientation = float(obstacle.orientation)
    cos_theta = math.cos(orientation)
    sin_theta = math.sin(orientation)
    along = abs(dx * cos_theta + dy * sin_theta)
    across = abs(-dx * sin_theta + dy * cos_theta)
    outside_length = max(along - float(obstacle.length) / 2.0, 0.0)
    outside_width = max(across - width / 2.0, 0.0)
    return math.hypot(outside_length, outside_width)
