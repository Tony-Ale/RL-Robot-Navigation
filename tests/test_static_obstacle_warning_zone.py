import math
import unittest
from types import SimpleNamespace

from custom_rewards.social_safety_reward import SocialSafetyRewardConfig, compute_social_safety_reward
from custom_rewards.socnavgym_social_safety_reward import Reward
from custom_rewards.static_obstacle_warning_zone import (
    StaticObstacleWarningZoneConfig,
    _surface_clearance,
    compute_static_obstacle_warning_zone,
    load_static_obstacle_warning_zone_config,
)
from training_pipeline.env_factory import validate_static_obstacle_reward_visibility


class DummyRobot(SimpleNamespace):
    def collides(self, _object):
        return False


class DummyEnv(SimpleNamespace):
    def get_waypoints(self):
        return []


def _robot(x=0.0, y=0.0, radius=0.2):
    return DummyRobot(x=x, y=y, radius=radius, goal_x=10.0, goal_y=0.0)


def _plant(x, y, radius=0.1):
    return SimpleNamespace(name="plant", x=x, y=y, radius=radius)


def _wall(x, y, orientation, length, thickness=0.2):
    return SimpleNamespace(
        name="wall",
        x=x,
        y=y,
        orientation=orientation,
        length=length,
        thickness=thickness,
    )


def _table(x, y, orientation, length, width):
    return SimpleNamespace(
        name="table",
        x=x,
        y=y,
        orientation=orientation,
        length=length,
        width=width,
    )


def _env(robot=None, walls=None, tables=None, plants=None, laptops=None):
    return DummyEnv(
        robot=robot or _robot(),
        shape="square",
        MAP_X=10.0,
        MAP_Y=10.0,
        walls=walls or [],
        tables=tables or [],
        plants=plants or [],
        laptops=laptops or [],
        static_humans=[],
        dynamic_humans=[],
        moving_interactions=[],
        static_interactions=[],
        h_l_interactions=[],
        ticks=0,
        EPISODE_LENGTH=100,
        GOAL_THRESHOLD=0.5,
    )


class TestStaticObstacleWarningZone(unittest.TestCase):
    def test_circle_clearance_uses_both_physical_radii(self):
        robot = _robot(x=0.0, radius=0.2)
        plant = _plant(x=0.8, y=0.0, radius=0.1)

        self.assertAlmostEqual(_surface_clearance(robot, plant), 0.5)

    def test_rotated_rectangle_clearance_uses_surface_not_centre(self):
        robot = _robot(x=0.5, y=0.0, radius=0.1)
        table = _table(x=0.0, y=0.0, orientation=math.pi / 2.0, length=2.0, width=0.4)

        self.assertAlmostEqual(_surface_clearance(robot, table), 0.2)

    def test_nearest_active_obstacle_sets_reward_without_summing_penalties(self):
        config = StaticObstacleWarningZoneConfig(
            enabled=True,
            warning_clearance=0.4,
            warning_zone_scale=0.2,
            include_walls=False,
            include_plants=True,
        )
        env = _env(plants=[_plant(0.7, 0.0), _plant(0.45, 0.0)])

        reward, info = compute_static_obstacle_warning_zone(env, config)

        expected_clearance = 0.15
        expected_reward = config.warning_zone_scale * math.expm1(expected_clearance - config.warning_clearance)
        self.assertAlmostEqual(info["nearest_static_clearance"], expected_clearance)
        self.assertEqual(info["nearest_static_type"], "plant")
        self.assertEqual(info["static_warning_zone_hits"], 2)
        self.assertAlmostEqual(reward, expected_reward)
        self.assertAlmostEqual(info["static_warning_zone_reward"], expected_reward)

    def test_penalty_is_zero_at_threshold_and_monotonic_when_approaching(self):
        config = StaticObstacleWarningZoneConfig(
            enabled=True,
            warning_clearance=0.4,
            warning_zone_scale=0.2,
            include_walls=False,
            include_plants=True,
        )
        rewards = []
        for clearance in (0.4, 0.2, 0.05):
            centre_distance = clearance + 0.2 + 0.1
            reward, _ = compute_static_obstacle_warning_zone(
                _env(plants=[_plant(centre_distance, 0.0)]),
                config,
            )
            rewards.append(reward)

        self.assertAlmostEqual(rewards[0], 0.0)
        self.assertGreater(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])

    def test_wall_reward_is_invariant_to_observation_segment_size(self):
        config = StaticObstacleWarningZoneConfig(
            enabled=True,
            warning_clearance=0.4,
            warning_zone_scale=0.2,
            include_walls=True,
            include_boundary_walls=False,
        )
        corridor = _wall(0.0, 0.0, 0.0, 4.0)
        env = _env(robot=_robot(y=0.4), walls=[corridor])

        env.wall_segment_size = 3.0
        first_reward, first_info = compute_static_obstacle_warning_zone(env, config)
        env.wall_segment_size = 15.0
        second_reward, second_info = compute_static_obstacle_warning_zone(env, config)

        self.assertAlmostEqual(first_reward, second_reward)
        self.assertAlmostEqual(
            first_info["nearest_static_clearance"],
            second_info["nearest_static_clearance"],
        )

    def test_boundary_filter_ignores_perimeter_wall(self):
        config = StaticObstacleWarningZoneConfig(
            enabled=True,
            warning_clearance=0.4,
            include_walls=True,
            include_boundary_walls=False,
        )
        boundary = _wall(0.0, 4.9, 0.0, 10.0)
        corridor = _wall(0.0, 0.0, 0.0, 4.0)
        env = _env(robot=_robot(y=4.5), walls=[boundary, corridor])

        reward, info = compute_static_obstacle_warning_zone(env, config)

        self.assertEqual(reward, 0.0)
        self.assertEqual(info["static_warning_zone_hits"], 0)
        self.assertEqual(info["nearest_static_type"], "wall")
        self.assertGreater(info["nearest_static_clearance"], config.warning_clearance)

    def test_invalid_config_values_fail_early(self):
        with self.assertRaisesRegex(ValueError, "warning_clearance"):
            load_static_obstacle_warning_zone_config({"warning_clearance": 0.0})
        with self.assertRaisesRegex(ValueError, "warning_zone_scale"):
            load_static_obstacle_warning_zone_config({"warning_zone_scale": -0.1})

    def test_policy_visibility_guard_rejects_unseen_static_types(self):
        config = {
            "architecture": {"entity_keys": ["humans"]},
            "wrappers": {"nearest_wall_segments": {"enabled": False}},
        }
        static_config = StaticObstacleWarningZoneConfig(enabled=True, include_walls=True)

        with self.assertRaisesRegex(ValueError, "walls"):
            validate_static_obstacle_reward_visibility(config, static_config)

    def test_policy_visibility_guard_accepts_matching_corridor_walls(self):
        config = {
            "architecture": {"entity_keys": ["humans", "walls"]},
            "wrappers": {
                "nearest_wall_segments": {
                    "enabled": True,
                    "observation_key": "walls",
                    "include_boundary_walls": False,
                }
            },
        }
        static_config = StaticObstacleWarningZoneConfig(
            enabled=True,
            include_walls=True,
            include_boundary_walls=False,
        )

        validate_static_obstacle_reward_visibility(config, static_config)

    def test_main_reward_composes_static_penalty_only_on_shaped_steps(self):
        static_config = StaticObstacleWarningZoneConfig(
            enabled=True,
            warning_clearance=0.4,
            warning_zone_scale=0.2,
            include_walls=False,
            include_plants=True,
        )
        config = SocialSafetyRewardConfig(
            goal_progress_scale=0.0,
            checkpoint_reward_enabled=False,
            static_obstacle_warning_zone=static_config,
        )
        env = _env(plants=[_plant(0.45, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=config,
            reached_goal=False,
            timeout=False,
            collision=False,
        )
        self.assertAlmostEqual(reward, info["static_warning_zone_reward"])
        self.assertLess(reward, 0.0)

        terminal_reward, terminal_info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=config,
            reached_goal=False,
            timeout=False,
            collision=True,
        )
        self.assertEqual(terminal_reward, config.object_collision_penalty)
        self.assertEqual(terminal_info["static_warning_zone_reward"], 0.0)

    def test_reward_adapter_keeps_static_penalty_out_of_distance_reward(self):
        adapter = Reward.__new__(Reward)
        adapter.info = {}
        reward_info = {
            "reward_reason": "shaped",
            "warning_zone_reward": -0.1,
            "static_warning_zone_reward": -0.2,
            "checkpoint_reward": 0.3,
            "stagnation_penalty": -0.02,
        }
        progress_reward = 0.4
        total_reward = progress_reward - 0.1 - 0.2 + 0.3 - 0.02

        adapter._update_info(total_reward, reward_info)

        self.assertAlmostEqual(adapter.info["distance_reward"], progress_reward)


if __name__ == "__main__":
    unittest.main()
