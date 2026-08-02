import unittest
from types import SimpleNamespace

import numpy as np

from custom_rewards.social_safety_reward import SocialSafetyRewardConfig
from custom_rewards.static_obstacle_warning_zone import (
    StaticObstacleWarningZoneConfig,
    compute_static_obstacle_warning_zone,
)
from custom_rewards.warning_zone_visualization_wrapper import WarningZoneVisualizationWrapper


class DummyRobot(SimpleNamespace):
    def collides(self, _object):
        return False


class DummyEnv:
    """Minimal environment object for testing the render callback without opening a GUI."""

    def __init__(self):
        self.robot = DummyRobot(x=0.5, y=0.0, goal_x=10.0, goal_y=0.0)
        self.dynamic_humans = [SimpleNamespace(x=0.0, y=0.0, orientation=0.0, width=0.72, speed=1.0)]
        self.static_humans = []
        self.moving_interactions = []
        self.static_interactions = []
        self.h_l_interactions = []
        self.walls = []
        self.tables = []
        self.plants = []
        self.laptops = []
        self.shape = "square"
        self.RESOLUTION_X = 200
        self.RESOLUTION_Y = 200
        self.MAP_X = 10.0
        self.MAP_Y = 10.0
        self.render_callbacks = []

    @property
    def unwrapped(self):
        return self


class TestWarningZoneVisualization(unittest.TestCase):
    """Tests for warning-zone rendering without calling cv2.imshow()."""

    def make_wrapper(self, enabled=True):
        """Create a visualization wrapper with deterministic colors and no GUI dependency."""
        print(f"Testing setup: visualizer enabled={enabled}")
        env = DummyEnv()
        cfg = {
            "visualization": {
                "enabled": enabled,
                "fill_alpha": 1.0,
                "draw_outline": False,
                "draw_heading_line": False,
                "draw_labels": False,
                "normal_color_bgr": [10, 20, 30],
                "active_color_bgr": [0, 0, 255],
            }
        }
        wrapper = WarningZoneVisualizationWrapper(env, config=cfg, reward_config=SocialSafetyRewardConfig())
        return wrapper

    def test_visualizer_installs_one_render_callback(self):
        """Testing: wrapper registers one SocNavGym render callback."""
        print("Testing: render callback installation")
        wrapper = self.make_wrapper()
        self.assertEqual(len(wrapper.unwrapped.render_callbacks), 1)

    def test_visualizer_draws_pixels_when_enabled(self):
        """Testing: enabled visualizer modifies a blank image using the reward warning-zone geometry."""
        print("Testing: enabled visualizer draws onto image")
        wrapper = self.make_wrapper(enabled=True)
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        wrapper.draw_warning_zones(image, wrapper.unwrapped)

        self.assertGreater(int(np.sum(image)), 0)

    def test_visualizer_does_not_draw_pixels_when_disabled(self):
        """Testing: disabled visualizer leaves the render image unchanged."""
        print("Testing: disabled visualizer leaves image unchanged")
        wrapper = self.make_wrapper(enabled=False)
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        wrapper.draw_warning_zones(image, wrapper.unwrapped)

        self.assertEqual(int(np.sum(image)), 0)

    def test_visualizer_uses_active_color_when_robot_is_inside_reward_trigger_zone(self):
        """Testing: active color appears when _warning_zone_contribution would trigger reward."""
        print("Testing: visualizer active color agrees with reward trigger")
        wrapper = self.make_wrapper(enabled=True)
        human = wrapper.unwrapped.dynamic_humans[0]

        self.assertEqual(wrapper._zone_color(human), tuple(wrapper.config["visualization"]["active_color_bgr"]))

    def test_visualizer_draws_static_footprint_where_reward_is_active(self):
        env = DummyEnv()
        env.dynamic_humans = []
        env.robot.radius = 0.2
        env.robot.x = 0.0
        env.robot.y = 0.45
        env.walls = [
            SimpleNamespace(
                name="wall",
                x=0.0,
                y=0.0,
                orientation=0.0,
                length=2.0,
                thickness=0.2,
            )
        ]
        static_config = StaticObstacleWarningZoneConfig(
            enabled=True,
            warning_clearance=0.25,
            include_walls=True,
            include_boundary_walls=False,
        )
        reward_config = SocialSafetyRewardConfig(static_obstacle_warning_zone=static_config)
        wrapper = WarningZoneVisualizationWrapper(
            env,
            config={
                "visualization": {
                    "enabled": True,
                    "fill_alpha": 1.0,
                    "draw_outline": False,
                    "draw_heading_line": False,
                    "draw_labels": False,
                }
            },
            reward_config=reward_config,
        )
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        reward, reward_info = compute_static_obstacle_warning_zone(env, static_config)
        wrapper.draw_warning_zones(image, env)

        self.assertLess(reward, 0.0)
        self.assertEqual(reward_info["static_warning_zone_hits"], 1)
        robot_pixel = wrapper._world_to_pixel(env.robot.x, env.robot.y)
        np.testing.assert_array_equal(
            image[robot_pixel[1], robot_pixel[0]],
            np.asarray(wrapper.config["visualization"]["static_active_color_bgr"], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
