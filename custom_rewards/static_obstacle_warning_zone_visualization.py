import math

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from custom_rewards.static_obstacle_warning_zone import _iter_static_obstacles, _surface_clearance


class StaticObstacleWarningZoneRenderer:
    """Draw the physical clearance zone around each static obstacle."""

    def __init__(self, env, reward_config, visualization_config, world_to_pixel):
        self.env = env
        self.config = reward_config.static_obstacle_warning_zone
        self.visualization_config = visualization_config
        self.world_to_pixel = world_to_pixel

    def fill(self, overlay):
        if not self.config.enabled:
            return
        for obstacle in _iter_static_obstacles(self.env, self.config):
            points = self._warning_polygon_pixels(obstacle)
            cv2.fillPoly(overlay, [points], self._zone_color(obstacle))

    def draw_details(self, image):
        if not self.config.enabled:
            return
        cfg = self.visualization_config
        for obstacle in _iter_static_obstacles(self.env, self.config):
            points = self._warning_polygon_pixels(obstacle)
            color = self._zone_color(obstacle)
            if cfg["draw_outline"]:
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=cfg["outline_thickness"])
            if cfg["draw_labels"]:
                clearance = _surface_clearance(self.env.robot, obstacle)
                label = f"{obstacle.name}: c={clearance:.2f}"
                cv2.putText(
                    image,
                    label,
                    self.world_to_pixel(obstacle.x, obstacle.y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def _warning_polygon_pixels(self, obstacle):
        # The polygon marks forbidden robot-centre positions, matching the
        # surface-clearance condition used by the reward function.
        expansion = float(self.config.warning_clearance) + float(self.env.robot.radius)
        geometry = self._obstacle_geometry(obstacle, expansion)
        return np.asarray(
            [self.world_to_pixel(x, y) for x, y in geometry.exterior.coords],
            dtype=np.int32,
        )

    def _obstacle_geometry(self, obstacle, expansion):
        resolution_degrees = max(float(self.visualization_config["sector_resolution_degrees"]), 0.5)
        resolution = max(4, int(math.ceil(90.0 / resolution_degrees)))
        if obstacle.name == "plant":
            return Point(float(obstacle.x), float(obstacle.y)).buffer(
                float(obstacle.radius) + expansion,
                quad_segs=resolution,
            )

        width = float(obstacle.thickness if obstacle.name == "wall" else obstacle.width)
        rectangle = Polygon(_rectangle_corners(obstacle, width))
        return rectangle.buffer(expansion, quad_segs=resolution)

    def _zone_color(self, obstacle):
        active = _surface_clearance(self.env.robot, obstacle) <= self.config.warning_clearance
        color_key = "static_active_color_bgr" if active else "static_normal_color_bgr"
        return tuple(self.visualization_config[color_key])


def _rectangle_corners(obstacle, width):
    orientation = float(obstacle.orientation)
    along_x = math.cos(orientation) * float(obstacle.length) / 2.0
    along_y = math.sin(orientation) * float(obstacle.length) / 2.0
    across_x = -math.sin(orientation) * width / 2.0
    across_y = math.cos(orientation) * width / 2.0
    x = float(obstacle.x)
    y = float(obstacle.y)
    return [
        (x + along_x + across_x, y + along_y + across_y),
        (x + along_x - across_x, y + along_y - across_y),
        (x - along_x - across_x, y - along_y - across_y),
        (x - along_x + across_x, y - along_y + across_y),
    ]
