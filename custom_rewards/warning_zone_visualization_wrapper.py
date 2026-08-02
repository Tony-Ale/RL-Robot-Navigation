import math
from pathlib import Path

import cv2
import gym
import numpy as np

from custom_rewards.social_safety_reward import (
    _dynamic_warning_angle,
    _dynamic_warning_radius,
    _iter_humans,
    _warning_zone_contribution,
    _warning_zone_heading,
    load_social_safety_reward_config,
)
from custom_rewards.static_obstacle_warning_zone_visualization import StaticObstacleWarningZoneRenderer


DEFAULT_CONFIG = {
    "visualization": {
        "enabled": True,
        "fill_alpha": 0.25,
        "draw_outline": True,
        "outline_thickness": 2,
        "draw_heading_line": True,
        "heading_line_scale": 0.8,
        "draw_labels": False,
        "sector_resolution_degrees": 4.0,
        "normal_color_bgr": [80, 180, 255],
        "active_color_bgr": [0, 0, 255],
        "static_normal_color_bgr": [255, 180, 80],
        "static_active_color_bgr": [0, 0, 255],
        "heading_color_bgr": [40, 40, 40],
    },
    "reward_config": {
        "path": "custom_rewards/social_safety_reward_config.yaml",
    },
}


class WarningZoneVisualizationWrapper(gym.Wrapper):
    """
    Draw warning zones for humans and enabled static obstacles on SocNavGym renders.

    This wrapper only adds a render callback. It does not change observations,
    rewards, actions, or environment dynamics.
    """

    def __init__(self, env, config_path=None, config=None, reward_config=None):
        super().__init__(env)
        self.config_path = Path(config_path).resolve() if config_path is not None else None
        self.config = self._load_config(config_path, config)
        self.reward_config = reward_config or self._load_reward_config()
        self._install_render_hook()

    def draw_warning_zones(self, image, env):
        if not self.config["visualization"]["enabled"]:
            return

        overlay = image.copy()
        for human in _iter_humans(self.unwrapped, self.reward_config):
            self._draw_human_warning_zone(overlay, image, human)
        static_renderer = self._static_renderer()
        static_renderer.fill(overlay)

        alpha = self.config["visualization"]["fill_alpha"]
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, dst=image)

        if self.config["visualization"]["draw_outline"] or self.config["visualization"]["draw_heading_line"] or self.config["visualization"]["draw_labels"]:
            for human in _iter_humans(self.unwrapped, self.reward_config):
                self._draw_human_warning_zone_details(image, human)
            static_renderer.draw_details(image)

    def _draw_human_warning_zone(self, overlay, image, human):
        points = self._sector_polygon_pixels(human)
        if points is None:
            return

        color = self._zone_color(human)
        cv2.fillPoly(overlay, [points], color)

    def _draw_human_warning_zone_details(self, image, human):
        points = self._sector_polygon_pixels(human)
        if points is None:
            return

        color = self._zone_color(human)
        cfg = self.config["visualization"]

        if cfg["draw_outline"]:
            cv2.polylines(image, [points], isClosed=True, color=color, thickness=cfg["outline_thickness"])

        if cfg["draw_heading_line"]:
            radius = _dynamic_warning_radius(human, self.reward_config)
            heading = self._visual_heading(human)
            start = self._world_to_pixel(human.x, human.y)
            end = self._world_to_pixel(
                human.x + radius * cfg["heading_line_scale"] * math.cos(heading),
                human.y + radius * cfg["heading_line_scale"] * math.sin(heading),
            )
            cv2.line(image, start, end, tuple(cfg["heading_color_bgr"]), 2)

        if cfg["draw_labels"]:
            radius = _dynamic_warning_radius(human, self.reward_config)
            angle = _dynamic_warning_angle(human, self.reward_config)
            label = f"r={radius:.2f}, a={math.degrees(min(angle, 2 * math.pi)):.0f}"
            cv2.putText(image, label, self._world_to_pixel(human.x, human.y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    def _sector_polygon_pixels(self, human):
        radius = _dynamic_warning_radius(human, self.reward_config)
        if radius <= 0.0:
            return None

        sector_angle = min(_dynamic_warning_angle(human, self.reward_config), 2.0 * math.pi)
        heading = self._visual_heading(human)
        start_angle = heading - sector_angle / 2.0
        end_angle = heading + sector_angle / 2.0

        resolution = math.radians(max(self.config["visualization"]["sector_resolution_degrees"], 0.5))
        steps = max(4, int(math.ceil(sector_angle / resolution)))

        points = [self._world_to_pixel(human.x, human.y)]
        for idx in range(steps + 1):
            angle = start_angle + (end_angle - start_angle) * idx / steps
            points.append(self._world_to_pixel(human.x + radius * math.cos(angle), human.y + radius * math.sin(angle)))
        return np.asarray(points, dtype=np.int32)

    def _zone_color(self, human):
        contribution = _warning_zone_contribution(self.unwrapped.robot, human, self.reward_config)
        if contribution is not None:
            return tuple(self.config["visualization"]["active_color_bgr"])
        return tuple(self.config["visualization"]["normal_color_bgr"])

    def _visual_heading(self, human):
        return _warning_zone_heading(human, self.reward_config)

    def _world_to_pixel(self, x, y):
        env = self.unwrapped
        pixel_to_world_x = env.RESOLUTION_X / env.MAP_X
        pixel_to_world_y = env.RESOLUTION_Y / env.MAP_Y
        px = int(pixel_to_world_x * (x + env.MAP_X / 2.0))
        py = int(pixel_to_world_y * (env.MAP_Y / 2.0 - y))
        return px, py

    def _static_renderer(self):
        return StaticObstacleWarningZoneRenderer(
            self.unwrapped,
            self.reward_config,
            self.config["visualization"],
            self._world_to_pixel,
        )

    def _install_render_hook(self):
        base_env = self.unwrapped
        callbacks = getattr(base_env, "render_callbacks", None)
        if callbacks is None:
            callbacks = []
            setattr(base_env, "render_callbacks", callbacks)
        callbacks.append(self.draw_warning_zones)

    def _load_reward_config(self):
        reward_path = Path(self.config["reward_config"]["path"])
        if not reward_path.is_absolute() and not reward_path.exists() and self.config_path is not None:
            reward_path = self.config_path.parent / reward_path
        return load_social_safety_reward_config(reward_path)

    def _load_config(self, config_path, config):
        merged = self._deep_copy(DEFAULT_CONFIG)
        if config_path is not None:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to load warning-zone visualization config files.") from exc
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            self._deep_update(merged, file_config)
        if config is not None:
            self._deep_update(merged, config)
        return merged

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
