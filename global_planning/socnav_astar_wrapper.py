import math
from dataclasses import dataclass

import cv2
import gym
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from global_planning.a_star import CostMapAStarPlanner, sample_path_by_distance
from navigation_features.waypoint_state import CURRENT_WAYPOINTS_ATTR


@dataclass
class AStarPlan:
    path_cells: list
    path_world: list
    waypoints: list
    cost: float


DEFAULT_CONFIG = {
    "grid": {
        "resolution": 0.1,
        "padding": 0.0,
    },
    "obstacles": {
        "inflation_radius": None,
        "safety_margin": 0.1,
        "include_walls": True,
        "include_tables": True,
        "include_laptops": True,
        "include_plants": True,
        "include_static_humans": True,
        "include_static_crowds": True,
        "include_human_laptop_interactions": True,
        "include_dynamic_humans": False,
        "include_dynamic_crowds": False,
    },
    "cost_map": {
        "enabled": True,
        "obstacle_cost_weight": 4.0,
        "obstacle_cost_decay": 0.6,
        "dynamic_human_cost_weight": 2.0,
        "dynamic_human_cost_decay": 0.8,
        "normalize": True,
    },
    "planner": {
        "allow_diagonal": True,
        "diagonal_corner_cutting": False,
        "heuristic_weight": 1.0,
        "snap_start_goal_to_free": True,
        "max_snap_distance": 1.0,
        "replan_on_reset": True,
        "replan_each_step": False,
    },
    "waypoints": {
        "interval": 0.5,
        "include_start": True,
        "include_goal": True,
    },
    "render": {
        "enabled": True,
        "draw_grid": False,
        "grid_alpha": 0.25,
        "draw_path": True,
        "draw_waypoints": True,
        "draw_checkpoint_radius": False,
        "checkpoint_radius": 0.3,
        "checkpoint_radius_color_bgr": [0, 180, 255],
        "checkpoint_radius_thickness": 1,
        "path_color_bgr": [255, 0, 255],
        "waypoint_color_bgr": [0, 255, 255],
        "start_color_bgr": [0, 255, 0],
        "goal_color_bgr": [0, 0, 255],
        "path_thickness": 2,
        "waypoint_radius": 4,
    },
}


class SocNavAStarWrapper(gym.Wrapper):
    """
    Builds an occupancy/cost grid from SocNavGym state and plans with A*.

    The wrapper keeps planning outside SocNavGym. The only SocNavGym source
    change it relies on is an optional render callback hook.
    """

    def __init__(self, env, config_path=None, config=None):
        super().__init__(env)
        self.config = self._load_config(config_path, config)
        self.static_grid = None
        self.dynamic_grid = None
        self.occupancy_grid = None
        self.cost_map = None
        self.grid_origin = None
        self.grid_shape = None
        self.latest_plan = None
        self.episode_astar_path_length = None
        self._install_render_hook()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_astar_path_length = None
        self.rebuild_static_grid()
        self.update_dynamic_grid()
        if self.config["planner"]["replan_on_reset"]:
            robot = self.unwrapped.robot
            start = (robot.x, robot.y)
            goal = (robot.goal_x, robot.goal_y)
            plan = self.plan_from_robot_to_goal()
            self.episode_astar_path_length = self._reference_path_length(plan.path_world, start, goal)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.config["planner"]["replan_each_step"]:
            self.update_dynamic_grid()
            self.plan_from_robot_to_goal()
        self._add_path_efficiency_metrics(info)
        return obs, reward, terminated, truncated, info

    def _add_path_efficiency_metrics(self, info):
        reference_length = self.episode_astar_path_length
        info["A_STAR_PATH_LENGTH"] = reference_length
        actual_length = info.get("PATH_LENGTH")
        if reference_length is None or actual_length is None:
            info["A_STAR_SPL"] = None
        elif not info.get("SUCCESS", False):
            info["A_STAR_SPL"] = 0.0
        elif reference_length <= 1e-9:
            info["A_STAR_SPL"] = 1.0 if float(actual_length) <= 1e-9 else 0.0
        else:
            info["A_STAR_SPL"] = float(reference_length) / max(float(reference_length), float(actual_length))

    @staticmethod
    def _path_length(points):
        return float(sum(math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:])))

    @classmethod
    def _reference_path_length(cls, points, start, goal):
        """Measure the grid path while accounting for its exact episode endpoints."""
        if not points:
            return None
        if len(points) == 1:
            return float(math.dist(start, goal))
        return float(math.dist(start, points[0]) + cls._path_length(points) + math.dist(points[-1], goal))

    def rebuild_static_grid(self):
        grid = self._empty_grid()
        for geom in self._static_obstacle_geometries():
            self._rasterize_geometry(grid, geom)
        self.static_grid = self._inflate_grid(grid)
        self._rebuild_combined_grid_and_cost_map()
        return self.static_grid

    def update_dynamic_grid(self):
        grid = self._empty_grid()
        if self.config["obstacles"]["include_dynamic_humans"]:
            for human in getattr(self.unwrapped, "dynamic_humans", []):
                self._rasterize_geometry(grid, self._circle_geometry(human.x, human.y, human.width / 2))
        if self.config["obstacles"]["include_dynamic_crowds"]:
            for crowd in getattr(self.unwrapped, "moving_interactions", []):
                self._rasterize_geometry(grid, self._interaction_geometry(crowd))
        self.dynamic_grid = self._inflate_grid(grid)
        self._rebuild_combined_grid_and_cost_map()
        return self.dynamic_grid

    def plan_from_robot_to_goal(self):
        robot = self.unwrapped.robot
        start = (robot.x, robot.y)
        goal = (robot.goal_x, robot.goal_y)
        return self.plan(start, goal)

    def plan(self, start_world, goal_world):
        if self.occupancy_grid is None:
            self.rebuild_static_grid()
            self.update_dynamic_grid()

        start_cell = self.world_to_grid(*start_world)
        goal_cell = self.world_to_grid(*goal_world)
        planner = CostMapAStarPlanner(
            self.occupancy_grid,
            cost_map=self.cost_map,
            allow_diagonal=self.config["planner"]["allow_diagonal"],
            diagonal_corner_cutting=self.config["planner"]["diagonal_corner_cutting"],
            heuristic_weight=self.config["planner"]["heuristic_weight"],
        )
        if self.config["planner"]["snap_start_goal_to_free"]:
            max_cells = int(math.ceil(self.config["planner"]["max_snap_distance"] / self.config["grid"]["resolution"]))
            start_cell = planner.nearest_free_cell(start_cell, max_cells) or start_cell
            goal_cell = planner.nearest_free_cell(goal_cell, max_cells) or goal_cell
        grid_path = planner.planning(start_cell, goal_cell)
        path_world = [self.grid_to_world(row, col) for row, col in grid_path.cells]
        waypoints = sample_path_by_distance(
            path_world,
            self.config["waypoints"]["interval"],
            include_start=self.config["waypoints"]["include_start"],
            include_goal=self.config["waypoints"]["include_goal"],
        )
        self.latest_plan = AStarPlan(grid_path.cells, path_world, waypoints, grid_path.cost)
        self._publish_current_waypoints(self.latest_plan.waypoints)
        return self.latest_plan

    def get_occupancy_grid(self, include_dynamic=True):
        if include_dynamic:
            return None if self.occupancy_grid is None else self.occupancy_grid.copy()
        return None if self.static_grid is None else self.static_grid.copy()

    def get_cost_map(self):
        return None if self.cost_map is None else self.cost_map.copy()

    def get_waypoints(self, interval=None):
        if self.latest_plan is None:
            return []
        if interval is None:
            return list(self.latest_plan.waypoints)
        return sample_path_by_distance(
            self.latest_plan.path_world,
            interval,
            include_start=self.config["waypoints"]["include_start"],
            include_goal=self.config["waypoints"]["include_goal"],
        )

    def _publish_current_waypoints(self, waypoints):
        setattr(self.unwrapped, CURRENT_WAYPOINTS_ATTR, list(waypoints))

    def world_to_grid(self, x, y):
        min_x, min_y = self.grid_origin
        resolution = self.config["grid"]["resolution"]
        col = int(round((x - min_x) / resolution))
        row = int(round((y - min_y) / resolution))
        height, width = self.grid_shape
        return max(0, min(height - 1, row)), max(0, min(width - 1, col))

    def grid_to_world(self, row, col):
        min_x, min_y = self.grid_origin
        resolution = self.config["grid"]["resolution"]
        return min_x + col * resolution, min_y + row * resolution

    def draw_astar_overlay(self, image, env):
        if not self.config["render"]["enabled"] or self.latest_plan is None:
            return

        if self.config["render"]["draw_grid"] and self.occupancy_grid is not None:
            self._draw_grid_overlay(image)

        if self.config["render"]["draw_path"] and self.latest_plan.path_world:
            self._draw_polyline(image, self.latest_plan.path_world)

        if self.config["render"]["draw_checkpoint_radius"]:
            self._draw_checkpoint_radius_overlay(image)

        if self.config["render"]["draw_waypoints"]:
            for point in self.latest_plan.waypoints:
                cv2.circle(
                    image,
                    self._world_to_pixel(point),
                    self.config["render"]["waypoint_radius"],
                    tuple(self.config["render"]["waypoint_color_bgr"]),
                    -1,
                )

        robot = self.unwrapped.robot
        cv2.circle(image, self._world_to_pixel((robot.x, robot.y)), 6, tuple(self.config["render"]["start_color_bgr"]), -1)
        cv2.circle(image, self._world_to_pixel((robot.goal_x, robot.goal_y)), 6, tuple(self.config["render"]["goal_color_bgr"]), -1)

    def _draw_checkpoint_radius_overlay(self, image):
        radius_m = float(self.config["render"]["checkpoint_radius"])
        if radius_m <= 0 or not self.latest_plan.waypoints:
            return
        radius_px = self._world_radius_to_pixel(radius_m)
        color = tuple(self.config["render"]["checkpoint_radius_color_bgr"])
        thickness = int(self.config["render"]["checkpoint_radius_thickness"])
        for point in self.latest_plan.waypoints:
            cv2.circle(image, self._world_to_pixel(point), radius_px, color, thickness)

    def _install_render_hook(self):
        base_env = self.unwrapped
        callbacks = getattr(base_env, "render_callbacks", None)
        if callbacks is None:
            callbacks = []
            setattr(base_env, "render_callbacks", callbacks)
        callbacks.append(self.draw_astar_overlay)

    def _load_config(self, config_path, config):
        merged = self._deep_copy(DEFAULT_CONFIG)
        if config_path is not None:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required to load A* wrapper YAML config files. Install it with: python -m pip install PyYAML") from exc
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f) or {}
            self._deep_update(merged, file_config)
        if config is not None:
            self._deep_update(merged, config)
        return merged

    def _empty_grid(self):
        env = self.unwrapped
        padding = self.config["grid"]["padding"]
        resolution = self.config["grid"]["resolution"]
        min_x = -env.MAP_X / 2 - padding
        max_x = env.MAP_X / 2 + padding
        min_y = -env.MAP_Y / 2 - padding
        max_y = env.MAP_Y / 2 + padding
        width = int(math.ceil((max_x - min_x) / resolution)) + 1 # +1 to include the last cell, that is to account for indexing.
        height = int(math.ceil((max_y - min_y) / resolution)) + 1
        self.grid_origin = (min_x, min_y)
        self.grid_shape = (height, width)
        return np.zeros((height, width), dtype=bool)

    def _static_obstacle_geometries(self):
        env = self.unwrapped
        config = self.config["obstacles"]
        geometries = []
        if config["include_walls"]:
            geometries.extend(self._object_geometry(wall) for wall in getattr(env, "walls", []))
        if config["include_tables"]:
            geometries.extend(self._object_geometry(table) for table in getattr(env, "tables", []))
        if config["include_laptops"]:
            geometries.extend(self._object_geometry(laptop) for laptop in getattr(env, "laptops", []))
        if config["include_plants"]:
            geometries.extend(self._object_geometry(plant) for plant in getattr(env, "plants", []))
        if config["include_static_humans"]:
            geometries.extend(self._object_geometry(human) for human in getattr(env, "static_humans", []))
        if config["include_static_crowds"]:
            geometries.extend(self._interaction_geometry(crowd) for crowd in getattr(env, "static_interactions", []))
        if config["include_human_laptop_interactions"]:
            geometries.extend(self._interaction_geometry(interaction) for interaction in getattr(env, "h_l_interactions", []))
        return [geom for geom in geometries if geom is not None and not geom.is_empty]

    def _object_geometry(self, obj):
        name = getattr(obj, "name", "")
        if name in ("wall", "table", "laptop"):
            thickness = getattr(obj, "thickness", None)
            width = getattr(obj, "width", thickness)
            return self._rectangle_geometry(obj.x, obj.y, obj.orientation, obj.length, width)
        if name == "plant":
            return self._circle_geometry(obj.x, obj.y, obj.radius)
        if name == "human":
            return self._circle_geometry(obj.x, obj.y, obj.width / 2)
        return None

    def _interaction_geometry(self, interaction):
        if getattr(interaction, "name", "") == "human-human-interaction":
            geoms = [self._object_geometry(human) for human in getattr(interaction, "humans", [])]
            geoms = [geom for geom in geoms if geom is not None]
            if not geoms:
                radius = getattr(interaction, "radius", None) or 0.72 # 0.72 is the default human diameter / human-human interaction radius in SocNavGym.
                return self._circle_geometry(interaction.x, interaction.y, radius)
            merged = geoms[0]
            for geom in geoms[1:]:
                merged = merged.union(geom)
            return merged
        if getattr(interaction, "name", "") == "human-laptop-interaction":
            geoms = [self._object_geometry(interaction.human), self._object_geometry(interaction.laptop)]
            geoms = [geom for geom in geoms if geom is not None]
            if not geoms:
                return None
            merged = geoms[0]
            for geom in geoms[1:]:
                merged = merged.union(geom)
            return merged
        return None

    @staticmethod
    def _rectangle_geometry(x, y, theta, length, width):
        # Construct the rectangle in its LOCAL coordinate frame, centered at the origin.
        # If length=L and width=W, then the four corners are simply (±L/2, ±W/2),
        # giving a rectangle centered at (0,0). We rotate these corners by θ using
        # the standard 2D rotation matrix:
        #
        #     [ cosθ  -sinθ ]
        # R = [ sinθ   cosθ ]
        #
        # which comes from rotating a point's polar angle φ to φ+θ:
        #
        #     x = r cosφ,  y = r sinφ
        #     => x' = x cosθ - y sinθ
        #     => y' = x sinθ + y cosθ
        #
        # The corners are stored as ROW vectors, so we multiply by Rᵀ
        # (corners @ R.T) rather than R. After rotation, we translate every
        # corner by (x, y) to move the rectangle from its local frame to its
        # actual world position.
        dx = length / 2
        dy = width / 2
        corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=float)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        points = corners @ rotation.T + np.array([x, y])
        return Polygon(points)

    @staticmethod
    def _circle_geometry(x, y, radius):
        return Point(x, y).buffer(radius)

    def _rasterize_geometry(self, grid, geometry):
        """Rasterize a continuous obstacle into the occupancy grid.
        Instead of checking the entire map, iterate only over the geometry's
        bounding box. For each candidate cell, test whether the cell's world
        point lies inside/intersects the geometry; if so, mark the cell occupied"""

        min_x, min_y, max_x, max_y = geometry.bounds
        min_row, min_col = self.world_to_grid(min_x, min_y)
        max_row, max_col = self.world_to_grid(max_x, max_y)
        prepared = prep(geometry)
        for row in range(min(min_row, max_row), max(min_row, max_row) + 1):
            for col in range(min(min_col, max_col), max(min_col, max_col) + 1):
                x, y = self.grid_to_world(row, col)
                if prepared.intersects(Point(x, y)):
                    grid[row, col] = True

    def _inflate_grid(self, grid):
        """
        Inflate obstacles in the occupancy grid to create a configuration-space
        representation. Cells within the robot clearance radius of an obstacle
        are marked occupied, ensuring planned paths maintain a safe distance
        from walls and other objects.
        """
        radius = self.config["obstacles"]["inflation_radius"]
        if radius is None:
            radius = getattr(self.unwrapped, "ROBOT_RADIUS", 0.0) + self.config["obstacles"]["safety_margin"]
        if radius <= 0:
            return grid
        cells = int(math.ceil(radius / self.config["grid"]["resolution"]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cells + 1, 2 * cells + 1))
        inflated = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
        return inflated.astype(bool)

    def _rebuild_combined_grid_and_cost_map(self):
        if self.static_grid is None:
            return
        if self.dynamic_grid is None:
            self.dynamic_grid = np.zeros_like(self.static_grid, dtype=bool)
        self.occupancy_grid = np.logical_or(self.static_grid, self.dynamic_grid)
        self.cost_map = self._build_cost_map()

    def _build_cost_map(self):
        if not self.config["cost_map"]["enabled"] or self.occupancy_grid is None:
            return np.ones_like(self.occupancy_grid, dtype=float) # if occupancy_grid is None, it still returns an array "array(1.0)""

        resolution = self.config["grid"]["resolution"]
        free = (~self.occupancy_grid).astype(np.uint8)
        distance = cv2.distanceTransform(free, cv2.DIST_L2, 5) * resolution
        decay = max(self.config["cost_map"]["obstacle_cost_decay"], resolution)
        weight = self.config["cost_map"]["obstacle_cost_weight"]
        cost = 1.0 + weight * np.exp(-distance / decay)
        cost[self.occupancy_grid] = math.inf

        if self.config["cost_map"]["normalize"]:
            finite = np.isfinite(cost)
            if np.any(finite):
                min_cost = np.min(cost[finite])
                cost[finite] = cost[finite] / max(min_cost, 1e-9)
        return cost

    def _draw_polyline(self, image, points):
        pixels = [self._world_to_pixel(point) for point in points]
        for start, end in zip(pixels, pixels[1:]):
            cv2.line(
                image,
                start,
                end,
                tuple(self.config["render"]["path_color_bgr"]),
                self.config["render"]["path_thickness"],
            )

    def _draw_grid_overlay(self, image):
        overlay = image.copy()
        occupied_rows, occupied_cols = np.where(self.occupancy_grid)
        for row, col in zip(occupied_rows, occupied_cols):
            cv2.circle(overlay, self._world_to_pixel(self.grid_to_world(row, col)), 1, (60, 60, 60), -1)
        alpha = self.config["render"]["grid_alpha"]
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)

    def _world_to_pixel(self, point):
        env = self.unwrapped
        x, y = point
        px = int(round((x + env.MAP_X / 2) * env.PIXEL_TO_WORLD_X))
        py = int(round((env.MAP_Y / 2 - y) * env.PIXEL_TO_WORLD_Y))
        return px, py

    def _world_radius_to_pixel(self, radius):
        env = self.unwrapped
        scale = (float(env.PIXEL_TO_WORLD_X) + float(env.PIXEL_TO_WORLD_Y)) / 2.0
        return max(1, int(round(radius * scale)))

    @staticmethod
    def _deep_copy(value):
        if isinstance(value, dict):
            return {k: SocNavAStarWrapper._deep_copy(v) for k, v in value.items()}
        if isinstance(value, list):
            return list(value)
        return value

    @staticmethod
    def _deep_update(target, update):
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                SocNavAStarWrapper._deep_update(target[key], value)
            else:
                target[key] = value
        return target
