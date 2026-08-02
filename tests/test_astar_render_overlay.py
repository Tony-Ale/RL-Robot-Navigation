import unittest
from types import SimpleNamespace

import gym
import numpy as np
from gym import spaces

from training_pipeline.utils import configure_matplotlib_cache

configure_matplotlib_cache()

from global_planning.a_star import sample_path_by_distance
from global_planning.socnav_astar_wrapper import AStarPlan, SocNavAStarWrapper
from navigation_features.waypoint_state import CURRENT_WAYPOINTS_ATTR


class DummyRenderEnv(gym.Env):
    """Minimal env for testing A* render overlays without SocNavGym."""

    MAP_X = 10.0
    MAP_Y = 10.0
    PIXEL_TO_WORLD_X = 10.0
    PIXEL_TO_WORLD_Y = 10.0

    def __init__(self):
        super().__init__()
        self.robot = SimpleNamespace(x=0.0, y=0.0, goal_x=2.0, goal_y=0.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    @property
    def unwrapped(self):
        return self


class TestAStarRenderOverlay(unittest.TestCase):
    """Tests for A* debug rendering."""

    def test_checkpoint_radius_overlay_draws_circle_around_waypoints(self):
        """Testing: checkpoint radius render overlay modifies pixels around waypoint circles."""
        print("Testing: A* render overlay draws checkpoint radius circles")
        wrapper = SocNavAStarWrapper(
            DummyRenderEnv(),
            config={
                "render": {
                    "enabled": True,
                    "draw_grid": False,
                    "draw_path": False,
                    "draw_waypoints": False,
                    "draw_checkpoint_radius": True,
                    "checkpoint_radius": 0.3,
                    "checkpoint_radius_color_bgr": [0, 180, 255],
                    "checkpoint_radius_thickness": 1,
                }
            },
        )
        wrapper.latest_plan = AStarPlan(
            path_cells=[],
            path_world=[(1.0, 0.0)],
            waypoints=[(1.0, 0.0)],
            cost=0.0,
        )
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        wrapper.draw_astar_overlay(image, wrapper.unwrapped)

        center_x, center_y = wrapper._world_to_pixel((1.0, 0.0))
        radius_px = wrapper._world_radius_to_pixel(0.3)
        circle_pixel = image[center_y, center_x + radius_px]
        np.testing.assert_array_equal(circle_pixel, np.array([0, 180, 255], dtype=np.uint8))

    def test_astar_wrapper_publishes_waypoints_to_unwrapped_env(self):
        """Testing: A* wrapper shares current waypoints with the base env for rewards."""
        print("Testing: A* wrapper publishes waypoints to base env")
        wrapper = SocNavAStarWrapper(DummyRenderEnv())

        wrapper._publish_current_waypoints([(1.0, 0.0), (2.0, 0.0)])

        self.assertEqual(
            getattr(wrapper.unwrapped, CURRENT_WAYPOINTS_ATTR),
            [(1.0, 0.0), (2.0, 0.0)],
        )

    def test_short_path_without_start_still_includes_goal_waypoint(self):
        """Testing: short paths do not crash when start waypoint is disabled."""
        print("Testing: short A* paths include the goal when start waypoint is disabled")
        path = [(0.0, 0.0), (0.5, 0.0)]

        waypoints = sample_path_by_distance(path, interval=1.5, include_start=False, include_goal=True)

        self.assertEqual(waypoints, [(0.5, 0.0)])

    def test_astar_path_efficiency_metrics_use_fixed_reference_length(self):
        """Testing: A*-referenced SPL uses geometric length and is zero on failure."""
        print("Testing: A* wrapper reports success-weighted path efficiency")
        wrapper = SocNavAStarWrapper(DummyRenderEnv())

        self.assertAlmostEqual(wrapper._path_length([(0.0, 0.0), (3.0, 4.0), (6.0, 4.0)]), 8.0)
        self.assertAlmostEqual(
            wrapper._reference_path_length(
                [(1.0, 0.0), (4.0, 4.0), (7.0, 4.0)],
                start=(0.0, 0.0),
                goal=(10.0, 4.0),
            ),
            12.0,
        )
        self.assertAlmostEqual(
            wrapper._reference_path_length([(1.0, 1.0)], start=(0.0, 0.0), goal=(3.0, 4.0)),
            5.0,
        )
        self.assertIsNone(wrapper._reference_path_length([], start=(0.0, 0.0), goal=(1.0, 1.0)))
        wrapper.episode_astar_path_length = 8.0

        success_info = {"SUCCESS": True, "PATH_LENGTH": 10.0}
        wrapper._add_path_efficiency_metrics(success_info)
        self.assertEqual(success_info["A_STAR_PATH_LENGTH"], 8.0)
        self.assertAlmostEqual(success_info["A_STAR_SPL"], 0.8)

        failure_info = {"SUCCESS": False, "PATH_LENGTH": 4.0}
        wrapper._add_path_efficiency_metrics(failure_info)
        self.assertEqual(failure_info["A_STAR_SPL"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
