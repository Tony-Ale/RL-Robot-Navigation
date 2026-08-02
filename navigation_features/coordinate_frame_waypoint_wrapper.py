import math

import cv2
import gym
import numpy as np
from gym import spaces

from navigation_features.waypoint_state import (
    LAST_REACHED_WAYPOINT_ATTR,
    PROGRESS_TARGET_DISTANCE_ATTR,
    PROGRESS_TARGET_SIGNATURE_ATTR,
    WAYPOINT_ADVANCE_RADIUS_ATTR,
    WAYPOINT_SIGNATURE_ATTR,
    advance_waypoint_index,
)

DEFAULT_CONFIG = {
    "coordinate_frame": {
        "mode": "heading_aligned",
        "transform_observations": True,
    },
    "waypoint_features": {
        "enabled": True,
        "observation_key": "waypoint_features",
        "max_waypoints": 2,
        "waypoint_interval": None,
        "replan_if_missing": True,
        "include_position": True,
        "include_distance": True,
        "include_bearing": True,
        "include_bearing_sin_cos": False,
        "advance_radius": 0.3,
        "skip_failed_reset_episodes": True,
        "max_reset_attempts": 20,
        "flatten": True,
        "pad_value": 0.0,
    },
    "visualization": {
        "enabled": False,
        "draw_waypoint_window": True,
        "draw_skipped_waypoints": True,
        "draw_active_waypoint": True,
        "slot_colors_bgr": [[0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]],
        "skipped_waypoint_color_bgr": [120, 120, 120],
        "window_waypoint_color_bgr": [255, 255, 0],
        "active_waypoint_color_bgr": [0, 0, 255],
        "active_waypoint_radius": 8,
        "skipped_waypoint_radius": 4,
        "waypoint_thickness": 2,
        "skipped_waypoint_thickness": 1,
        "draw_slot_labels": True,
        "label_color_bgr": [255, 255, 255],
    },
}


class CoordinateFrameWaypointWrapper(gym.Wrapper):
    """
    Adds robot-centric sequential waypoint features to SocNavGym observations.

    The wrapper is intended to sit after SocNavAStarWrapper:

        env = SocNavAStarWrapper(env, ...)
        env = CoordinateFrameWaypointWrapper(env, ...)

    SocNavGym observations are already robot-centric in the robot-heading frame.
    When goal_aligned mode is selected, this wrapper rotates positions and
    orientation vectors so +x points from the robot toward its goal.
    """

    ENTITY_FEATURE_DIM = 14
    ENTITY_KEYS = ("humans", "laptops", "tables", "plants", "walls")

    def __init__(self, env, config_path=None, config=None):
        super().__init__(env)
        self.config = self._load_config(config_path, config)
        self._validate_config()
        self._waypoint_window_signature = None
        self._last_reached_waypoint_index = -1
        self.observation_space = self._build_observation_space()
        self._install_render_hook()

    def reset(self, **kwargs):
        cfg = self.config["waypoint_features"]
        max_attempts = cfg["max_reset_attempts"] if cfg["enabled"] and cfg["skip_failed_reset_episodes"] else 1
        seeded_reset = kwargs.get("seed") is not None

        for attempt in range(max_attempts):
            obs, info = self.env.reset(**kwargs)
            self._reset_waypoint_window_state()
            waypoints = [] if not cfg["enabled"] else self._get_waypoints()
            if not cfg["enabled"] or waypoints:
                info = dict(info)
                if attempt > 0:
                    info["planner_failed_reset_attempts"] = attempt
                return self._process_observation(obs, waypoints=waypoints), info
            if seeded_reset:
                raise RuntimeError("Planner produced no waypoints after 1 reset attempt(s) with unchanged reset arguments.")
            print(f"No waypoints generated on reset attempt {attempt + 1}/{max_attempts}; resetting env.")

        raise RuntimeError(
            "Planner produced no waypoints after "
            f"{max_attempts} reset attempt(s) with unchanged reset arguments."
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._advance_waypoint_window_state()
        return self._process_observation(obs), reward, terminated, truncated, info

    def get_waypoint_features(self):
        """Return only the current waypoint feature vector."""
        return self._build_waypoint_features()

    def _process_observation(self, obs, waypoints=None):
        if not isinstance(obs, dict):
            return obs

        processed = {key: np.array(value, copy=True) for key, value in obs.items()}
        goal_angle = self._goal_angle_from_observation(processed)

        if self.config["coordinate_frame"]["transform_observations"]:
            processed = self._transform_observation_frame(processed, goal_angle)

        if self.config["waypoint_features"]["enabled"]:
            key = self.config["waypoint_features"]["observation_key"]
            processed[key] = self._build_waypoint_features(goal_angle, waypoints=waypoints)

        return processed

    def _transform_observation_frame(self, obs, goal_angle):
        if self.config["coordinate_frame"]["mode"] == "heading_aligned":
            return obs

        if "robot" in obs and obs["robot"].shape[0] >= 8:
            # started from index 6 because from 0 to 5 is the robots one hot encoding.
            obs["robot"][6:8] = self._rotate_xy(obs["robot"][6], obs["robot"][7], goal_angle)

        for key in self.ENTITY_KEYS:
            if key in obs:
                obs[key] = self._transform_entity_array(obs[key], goal_angle)

        return obs

    def _transform_entity_array(self, values, goal_angle):
        if values.size == 0 or values.size % self.ENTITY_FEATURE_DIM != 0:
            return values

        entities = values.reshape(-1, self.ENTITY_FEATURE_DIM)
        entities[:, 6], entities[:, 7] = self._rotate_xy(entities[:, 6], entities[:, 7], goal_angle)

        sin_theta = entities[:, 8].copy()
        cos_theta = entities[:, 9].copy()
        c = math.cos(goal_angle)
        s = math.sin(goal_angle)
        entities[:, 8] = sin_theta * c - cos_theta * s
        entities[:, 9] = cos_theta * c + sin_theta * s
        return entities.reshape(values.shape)

    def _build_waypoint_features(self, goal_angle=None, waypoints=None):
        cfg = self.config["waypoint_features"]
        waypoint_dim = self._waypoint_feature_dim()
        shape = self._waypoint_feature_shape()
        features = np.full(shape, cfg["pad_value"], dtype=np.float32)

        if not cfg["enabled"]:
            return features

        if waypoints is None:
            waypoints = self._get_waypoints()
        waypoints = self._waypoints_for_observation_window(waypoints)
        if not waypoints:
            return features

        if goal_angle is None:
            goal_angle = self._goal_angle_from_robot_state()

        rows = []
        for waypoint in waypoints:
            x_frame, y_frame = self._world_to_active_frame(waypoint, goal_angle)
            distance = math.hypot(x_frame, y_frame)
            bearing = math.atan2(y_frame, x_frame)

            row = []
            if cfg["include_position"]:
                row.extend([x_frame, y_frame])
            if cfg["include_distance"]:
                row.append(distance)
            if cfg["include_bearing"]:
                row.append(bearing)
            if cfg["include_bearing_sin_cos"]:
                row.extend([math.sin(bearing), math.cos(bearing)])
            rows.append(row)

        waypoint_array = np.asarray(rows, dtype=np.float32)
        if cfg["flatten"]:
            flat = features.reshape(cfg["max_waypoints"], waypoint_dim)
            flat[: waypoint_array.shape[0], :] = waypoint_array
            return flat.reshape(-1)

        features[: waypoint_array.shape[0], :] = waypoint_array
        return features

    def _get_waypoints(self):
        cfg = self.config["waypoint_features"]

        if getattr(self.env, "latest_plan", None) is None and cfg["replan_if_missing"] and hasattr(self.env, "plan_from_robot_to_goal"):
            self.env.plan_from_robot_to_goal()

        interval = cfg["waypoint_interval"]
        if hasattr(self.env, "get_waypoints"):
            return self.env.get_waypoints(interval=interval)

        plan = getattr(self.env, "latest_plan", None)
        if plan is None:
            return []
        return list(getattr(plan, "waypoints", []))

    def _current_waypoint_window(self, waypoints):
        start = self._waypoint_window_start_index(waypoints)
        if start is None:
            return []
        return waypoints[start:]

    def _waypoint_window_start_index(self, waypoints):
        if not waypoints:
            self._reset_waypoint_window_state()
            return None

        self._sync_waypoint_signature(waypoints)
        self._sync_with_shared_waypoint_state()
        return min(self._last_reached_waypoint_index + 1, len(waypoints))

    def _waypoints_for_observation_window(self, waypoints):
        waypoints = self._current_waypoint_window(waypoints)
        return self._fixed_size_waypoint_window(waypoints)

    def _fixed_size_waypoint_window(self, waypoints):
        if not waypoints:
            return []

        cfg = self.config["waypoint_features"]
        window = list(waypoints[: cfg["max_waypoints"]])
        while len(window) < cfg["max_waypoints"]:
            window.append(window[-1])
        return window

    def _waypoints_for_visualization(self):
        waypoints = self._get_waypoints()
        start = self._waypoint_window_start_index(waypoints)
        if start is None:
            return [], []
        return list(waypoints[:start]), self._fixed_size_waypoint_window(waypoints[start:])

    def _advance_waypoint_window_state(self):
        waypoints = self._get_waypoints()
        if not waypoints:
            self._reset_waypoint_window_state()
            return

        self._sync_waypoint_signature(waypoints)
        self._sync_with_shared_waypoint_state()
        self._advance_reached_waypoint_index(waypoints)
        self._publish_shared_waypoint_state()

    def _sync_waypoint_signature(self, waypoints):
        signature = self._waypoint_signature(waypoints)
        if self._waypoint_window_signature != signature:
            self._waypoint_window_signature = signature
            self._last_reached_waypoint_index = -1
            self._clear_shared_progress_target_state()
            self._publish_shared_waypoint_state()

    def _advance_reached_waypoint_index(self, waypoints):
        radius = self._waypoint_advance_radius()
        if radius <= 0:
            return

        robot = self.unwrapped.robot
        self._last_reached_waypoint_index, _hits = advance_waypoint_index(
            robot.x,
            robot.y,
            waypoints,
            self._last_reached_waypoint_index,
            radius,
        )

    def _sync_with_shared_waypoint_state(self):
        shared_index = getattr(self.unwrapped, LAST_REACHED_WAYPOINT_ATTR, None)
        if shared_index is not None:
            self._last_reached_waypoint_index = max(self._last_reached_waypoint_index, int(shared_index))

    def _publish_shared_waypoint_state(self):
        setattr(self.unwrapped, WAYPOINT_SIGNATURE_ATTR, self._waypoint_window_signature)
        setattr(self.unwrapped, LAST_REACHED_WAYPOINT_ATTR, self._last_reached_waypoint_index)
        setattr(self.unwrapped, WAYPOINT_ADVANCE_RADIUS_ATTR, self._waypoint_advance_radius())

    def _clear_shared_progress_target_state(self):
        if hasattr(self.unwrapped, PROGRESS_TARGET_SIGNATURE_ATTR):
            delattr(self.unwrapped, PROGRESS_TARGET_SIGNATURE_ATTR)
        if hasattr(self.unwrapped, PROGRESS_TARGET_DISTANCE_ATTR):
            delattr(self.unwrapped, PROGRESS_TARGET_DISTANCE_ATTR)

    def _waypoint_advance_radius(self):
        return float(self.config["waypoint_features"]["advance_radius"])

    def _waypoint_signature(self, waypoints):
        return tuple((round(float(x), 4), round(float(y), 4)) for x, y in waypoints)

    def _reset_waypoint_window_state(self):
        self._waypoint_window_signature = None
        self._last_reached_waypoint_index = -1
        self._clear_shared_progress_target_state()
        self._publish_shared_waypoint_state()

    def _world_to_active_frame(self, waypoint, goal_angle):
        robot = self.unwrapped.robot
        dx = waypoint[0] - robot.x
        dy = waypoint[1] - robot.y

        heading = getattr(robot, "orientation", 0.0)
        c = math.cos(heading)
        s = math.sin(heading)
        x_heading = c * dx + s * dy
        y_heading = -s * dx + c * dy

        if self.config["coordinate_frame"]["mode"] == "heading_aligned":
            return x_heading, y_heading
        return self._rotate_xy(x_heading, y_heading, goal_angle)

    def _goal_angle_from_observation(self, obs):
        robot_obs = obs.get("robot")
        if robot_obs is None or robot_obs.shape[0] < 8:
            return 0.0

        goal_x = float(robot_obs[6])
        goal_y = float(robot_obs[7])
        if math.hypot(goal_x, goal_y) < 1e-8:
            return 0.0
        return math.atan2(goal_y, goal_x)

    def _goal_angle_from_robot_state(self):
        """computes goal angle from the simulator/world state"""
        robot = self.unwrapped.robot
        dx = robot.goal_x - robot.x
        dy = robot.goal_y - robot.y
        heading = getattr(robot, "orientation", 0.0)

        c = math.cos(heading)
        s = math.sin(heading)
        goal_x_heading = c * dx + s * dy
        goal_y_heading = -s * dx + c * dy
        if math.hypot(goal_x_heading, goal_y_heading) < 1e-8:
            return 0.0
        return math.atan2(goal_y_heading, goal_x_heading)

    def _rotate_xy(self, x, y, angle):
        """Rotate a point by the given angle """
        c = math.cos(angle)
        s = math.sin(angle)
        return c * x + s * y, -s * x + c * y

    def draw_waypoint_window_overlay(self, image, _env):
        viz_cfg = self.config["visualization"]
        if not viz_cfg["enabled"] or not viz_cfg["draw_waypoint_window"] or not self.config["waypoint_features"]["enabled"]:
            return

        skipped_waypoints, window_waypoints = self._waypoints_for_visualization()
        if not skipped_waypoints and not window_waypoints:
            return

        if viz_cfg["draw_skipped_waypoints"]:
            for waypoint in skipped_waypoints:
                cv2.circle(
                    image,
                    self._world_to_pixel(waypoint),
                    int(viz_cfg["skipped_waypoint_radius"]),
                    tuple(viz_cfg["skipped_waypoint_color_bgr"]),
                    int(viz_cfg["skipped_waypoint_thickness"]),
                )

        for index in reversed(range(len(window_waypoints))):
            waypoint = window_waypoints[index]
            color = self._waypoint_slot_color(index)
            radius = self._waypoint_slot_radius(index)
            pixel = self._world_to_pixel(waypoint)
            cv2.circle(image, pixel, radius, color, int(viz_cfg["waypoint_thickness"]))
            if viz_cfg["draw_slot_labels"]:
                cv2.putText(
                    image,
                    str(index + 1),
                    (pixel[0] + radius + 2, pixel[1] - radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    tuple(viz_cfg["label_color_bgr"]),
                    1,
                    cv2.LINE_AA,
                )

    def _waypoint_slot_color(self, index):
        viz_cfg = self.config["visualization"]
        colors = viz_cfg.get("slot_colors_bgr") or []
        if index < len(colors):
            return tuple(colors[index])
        if index == 0 and viz_cfg["draw_active_waypoint"]:
            return tuple(viz_cfg["active_waypoint_color_bgr"])
        return tuple(viz_cfg["window_waypoint_color_bgr"])

    def _waypoint_slot_radius(self, index):
        viz_cfg = self.config["visualization"]
        if index == 0:
            return int(viz_cfg["active_waypoint_radius"])
        return int(viz_cfg["active_waypoint_radius"]) + index * max(1, int(viz_cfg["waypoint_thickness"]))

    def _install_render_hook(self):
        if not self.config["visualization"]["enabled"]:
            return
        callbacks = getattr(self.unwrapped, "render_callbacks", None)
        if callbacks is None:
            callbacks = []
            setattr(self.unwrapped, "render_callbacks", callbacks)
        callbacks.append(self.draw_waypoint_window_overlay)

    def _world_to_pixel(self, point):
        env = self.unwrapped
        x, y = point
        px = int(round((x + env.MAP_X / 2) * env.PIXEL_TO_WORLD_X))
        py = int(round((env.MAP_Y / 2 - y) * env.PIXEL_TO_WORLD_Y))
        return px, py

    def _build_observation_space(self):
        if not isinstance(self.env.observation_space, spaces.Dict):
            return self.env.observation_space

        observation_spaces = dict(self.env.observation_space.spaces)
        if self.config["waypoint_features"]["enabled"]:
            key = self.config["waypoint_features"]["observation_key"]
            shape = self._waypoint_feature_shape()
            observation_spaces[key] = spaces.Box(
                low=np.full(shape, -np.inf, dtype=np.float32),
                high=np.full(shape, np.inf, dtype=np.float32),
                dtype=np.float32,
            )
        return spaces.Dict(observation_spaces)

    def _waypoint_feature_shape(self):
        cfg = self.config["waypoint_features"]
        max_waypoints = cfg["max_waypoints"]
        feature_dim = self._waypoint_feature_dim()
        if cfg["flatten"]:
            return (max_waypoints * feature_dim,)
        return (max_waypoints, feature_dim)

    def _waypoint_feature_dim(self):
        cfg = self.config["waypoint_features"]
        dim = 0
        if cfg["include_position"]:
            dim += 2
        if cfg["include_distance"]:
            dim += 1
        if cfg["include_bearing"]:
            dim += 1
        if cfg["include_bearing_sin_cos"]:
            dim += 2
        return dim

    def _load_config(self, config_path, config):
        merged = self._deep_copy(DEFAULT_CONFIG)
        if config_path is not None:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to load navigation feature YAML config files.") from exc
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            self._deep_update(merged, file_config)
        if config is not None:
            self._deep_update(merged, config)
        return merged

    def _validate_config(self):
        mode = self.config["coordinate_frame"]["mode"]
        if mode not in {"heading_aligned", "goal_aligned"}:
            raise ValueError('coordinate_frame.mode must be "heading_aligned" or "goal_aligned".')

        cfg = self.config["waypoint_features"]
        if cfg["max_waypoints"] <= 0:
            raise ValueError("waypoint_features.max_waypoints must be greater than zero.")
        if cfg["advance_radius"] < 0:
            raise ValueError("waypoint_features.advance_radius must be greater than or equal to zero.")
        if cfg["max_reset_attempts"] <= 0:
            raise ValueError("waypoint_features.max_reset_attempts must be greater than zero.")
        if self._waypoint_feature_dim() <= 0:
            raise ValueError("At least one waypoint feature component must be enabled.")

    def _deep_copy(self, value):
        if isinstance(value, dict):
            return {key: self._deep_copy(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._deep_copy(item) for item in value]
        return value

    def _deep_update(self, base, updates):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
