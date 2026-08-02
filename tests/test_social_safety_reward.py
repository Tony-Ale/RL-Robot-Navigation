import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from custom_rewards.social_safety_reward import (
    SocialSafetyRewardConfig,
    _dynamic_warning_angle,
    _dynamic_warning_radius,
    _robot_inside_human_warning_sector,
    _warning_zone_activation_radius,
    compute_social_safety_reward,
    load_social_safety_reward_config,
    reset_stagnation_tracking,
)
from custom_rewards.socnavgym_social_safety_reward import Reward
from navigation_features.waypoint_state import CURRENT_WAYPOINTS_ATTR, WAYPOINT_ADVANCE_RADIUS_ATTR


class DummyRobot(SimpleNamespace):
    def collides(self, _object):
        return False


class SelectiveCollisionRobot(DummyRobot):
    def __init__(self, colliding_names, **kwargs):
        super().__init__(**kwargs)
        self.colliding_names = set(colliding_names)

    def collides(self, obj):
        return getattr(obj, "name", None) in self.colliding_names


def make_human(x=0.0, y=0.0, orientation=0.0, width=0.72, speed=0.0):
    return SimpleNamespace(name="human", x=x, y=y, orientation=orientation, width=width, speed=speed)


class DummyRewardEnv(SimpleNamespace):
    def get_waypoints(self):
        return getattr(self, "waypoints", [])


class DummySocNavRewardEnv(DummyRewardEnv):
    def __init__(self):
        super().__init__(
            robot=DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0, radius=0.25),
            ticks=0,
            EPISODE_LENGTH=100,
            GOAL_THRESHOLD=0.5,
            static_humans=[],
            dynamic_humans=[],
            plants=[],
            walls=[],
            tables=[],
            laptops=[],
            moving_interactions=[],
            static_interactions=[],
            h_l_interactions=[],
            waypoints=[],
        )


def make_env(robot, dynamic_humans=None, waypoints=None):
    return DummyRewardEnv(
        robot=robot,
        ticks=0,
        EPISODE_LENGTH=100,
        GOAL_THRESHOLD=0.5,
        static_humans=[],
        dynamic_humans=dynamic_humans or [],
        plants=[],
        walls=[],
        tables=[],
        laptops=[],
        moving_interactions=[],
        static_interactions=[],
        h_l_interactions=[],
        waypoints=waypoints or [],
    )


class TestSocialSafetyReward(unittest.TestCase):
    """Tests for the custom social-safety reward and dynamic warning-zone math."""

    def test_project_reward_config_uses_radius_sized_gait_width_scale(self):
        """Testing: project config uses human-radius proxy for gait width."""
        print("Testing: project reward config gait width scale")
        cfg = load_social_safety_reward_config("custom_rewards/social_safety_reward_config.yaml")

        self.assertAlmostEqual(cfg.gait_width_scale, 0.7)

    def test_warning_radius_and_activation_radius_use_speed_width_and_robot_radius(self):
        """Testing: r_wz uses speed/width and trigger radius adds robot radius."""
        print("Testing: warning radius and activation radius from speed, width, and robot radius")
        cfg = SocialSafetyRewardConfig()
        human = make_human(width=0.72, speed=0.5)
        robot = DummyRobot(radius=0.25)

        expected_warning_radius = 0.8 * 0.5 + 0.5 * 0.72
        expected_activation_radius = expected_warning_radius + 0.25

        self.assertAlmostEqual(_dynamic_warning_radius(human, cfg), expected_warning_radius)
        self.assertAlmostEqual(_warning_zone_activation_radius(human, robot, cfg), expected_activation_radius)

    def test_static_human_activation_zone_includes_robot_radius(self):
        """Testing: static humans trigger before body collision when robot radius is included."""
        print("Testing: static human baseline warning trigger radius")
        cfg = SocialSafetyRewardConfig()
        human = make_human(width=0.72, speed=0.0)
        robot = DummyRobot(radius=0.25)

        self.assertAlmostEqual(_dynamic_warning_radius(human, cfg), 0.36)
        self.assertAlmostEqual(_warning_zone_activation_radius(human, robot, cfg), 0.61)

    def test_warning_sector_contains_robot_in_front_and_rejects_robot_behind_when_speed_is_high(self):
        """Testing: the sector is centered on the human heading direction."""
        print("Testing: sector direction accepts robot in front and rejects robot behind")
        cfg = SocialSafetyRewardConfig()
        human = make_human(orientation=0.0, speed=1.0)
        robot_in_front = DummyRobot(x=0.5, y=0.0, goal_x=10.0, goal_y=0.0)
        robot_behind = DummyRobot(x=-0.5, y=0.0, goal_x=10.0, goal_y=0.0)

        self.assertLess(_dynamic_warning_angle(human, cfg), 2.0 * math.pi)
        self.assertTrue(_robot_inside_human_warning_sector(robot_in_front, human, cfg))
        self.assertFalse(_robot_inside_human_warning_sector(robot_behind, human, cfg))

    def test_reward_returns_terminal_rewards_before_warning_or_progress_rewards(self):
        """Testing: goal, timeout, and collision branches have priority."""
        print("Testing: terminal reward branch priority")
        cfg = SocialSafetyRewardConfig()
        robot = DummyRobot(x=0.0, y=0.0, goal_x=5.0, goal_y=0.0)
        env = make_env(robot, dynamic_humans=[make_human(x=0.2, y=0.0, speed=1.0)])

        reward, info = compute_social_safety_reward(env, previous_goal_distance=5.0, config=cfg, reached_goal=True)
        self.assertEqual(reward, cfg.goal_reward)
        self.assertEqual(info["reward_reason"], "goal")

        reward, info = compute_social_safety_reward(env, previous_goal_distance=5.0, config=cfg, timeout=True)
        self.assertEqual(reward, cfg.timeout_penalty)
        self.assertEqual(info["reward_reason"], "timeout")

        reward, info = compute_social_safety_reward(env, previous_goal_distance=5.0, config=cfg, collision=True)
        self.assertEqual(reward, cfg.object_collision_penalty)
        self.assertEqual(info["reward_reason"], "collision")

    def test_human_collision_uses_human_collision_penalty(self):
        """Testing: human collisions use the human-specific collision penalty."""
        print("Testing: human collision penalty branch")
        cfg = SocialSafetyRewardConfig(human_collision_penalty=-2.0, object_collision_penalty=-0.2)
        robot = SelectiveCollisionRobot({"human"}, x=0.0, y=0.0, goal_x=5.0, goal_y=0.0)
        env = make_env(robot, dynamic_humans=[make_human(x=0.0, y=0.0)])

        reward, info = compute_social_safety_reward(env, previous_goal_distance=5.0, config=cfg)

        self.assertEqual(reward, cfg.human_collision_penalty)
        self.assertEqual(info["reward_reason"], "human_collision")

    def test_object_collision_uses_object_collision_penalty(self):
        """Testing: object collisions use the object-specific collision penalty."""
        print("Testing: object collision penalty branch")
        cfg = SocialSafetyRewardConfig(human_collision_penalty=-2.0, object_collision_penalty=-0.2)
        robot = SelectiveCollisionRobot({"plant"}, x=0.0, y=0.0, goal_x=5.0, goal_y=0.0)
        env = make_env(robot)
        env.plants = [SimpleNamespace(name="plant")]

        reward, info = compute_social_safety_reward(env, previous_goal_distance=5.0, config=cfg)

        self.assertEqual(reward, cfg.object_collision_penalty)
        self.assertEqual(info["reward_reason"], "object_collision")

    def test_human_collision_takes_priority_over_object_collision(self):
        """Testing: simultaneous human/object collisions return the human penalty."""
        print("Testing: human collision penalty has priority over object collision")
        cfg = SocialSafetyRewardConfig(human_collision_penalty=-2.0, object_collision_penalty=-0.2)
        robot = SelectiveCollisionRobot({"human", "plant"}, x=0.0, y=0.0, goal_x=5.0, goal_y=0.0)
        env = make_env(robot, dynamic_humans=[make_human(x=0.0, y=0.0)])
        env.plants = [SimpleNamespace(name="plant")]

        reward, info = compute_social_safety_reward(env, previous_goal_distance=5.0, config=cfg)

        self.assertEqual(reward, cfg.human_collision_penalty)
        self.assertEqual(info["reward_reason"], "human_collision")

    def test_reward_returns_warning_zone_reward_when_robot_is_inside_active_sector(self):
        """Testing: warning-zone branch activates when the robot is inside the trigger sector."""
        print("Testing: warning-zone reward branch")
        cfg = SocialSafetyRewardConfig()
        human = make_human(x=0.0, y=0.0, orientation=0.0, speed=1.0)
        robot = DummyRobot(x=0.5, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, dynamic_humans=[human])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["warning_zone_hits"], 1)
        self.assertAlmostEqual(reward, info["warning_zone_reward"] + cfg.goal_progress_scale * 0.5)
        self.assertLessEqual(reward, 0.0)

    def test_reward_returns_global_goal_progress_when_no_waypoints_are_available(self):
        """Testing: progress shaping falls back to the global goal when no waypoints exist."""
        print("Testing: global-goal progress fallback branch")
        cfg = SocialSafetyRewardConfig()
        robot = DummyRobot(x=2.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, dynamic_humans=[make_human(x=0.0, y=5.0, speed=1.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=9.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["progress_target"], "goal")
        self.assertIsNone(info["progress_target_index"])
        self.assertAlmostEqual(reward, cfg.goal_progress_scale * (9.0 - 8.0))

    def test_stagnation_penalty_applies_after_windowed_low_displacement(self):
        """Testing: stagnation penalty activates only after sustained low displacement."""
        print("Testing: stagnation penalty activates after displacement window")
        cfg = SocialSafetyRewardConfig(
            goal_progress_scale=0.0,
            checkpoint_reward_enabled=False,
            stagnation_penalty_enabled=True,
            stagnation_window_steps=12,
            stagnation_min_displacement=0.1,
            stagnation_penalty=-0.02,
        )
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot)
        reset_stagnation_tracking(env)

        for tick in range(1, 12):
            env.ticks = tick
            reward, info = compute_social_safety_reward(
                env,
                previous_goal_distance=10.0,
                config=cfg,
                reached_goal=False,
                timeout=False,
                collision=False,
            )
            self.assertAlmostEqual(reward, 0.0)
            self.assertFalse(info["stagnation_stalled"])

        env.ticks = 12
        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertAlmostEqual(reward, cfg.stagnation_penalty)
        self.assertAlmostEqual(info["stagnation_penalty"], cfg.stagnation_penalty)
        self.assertTrue(info["stagnation_stalled"])
        self.assertAlmostEqual(info["stagnation_displacement"], 0.0)

    def test_stagnation_penalty_clears_after_sufficient_displacement(self):
        """Testing: moving beyond the displacement threshold clears stagnation."""
        print("Testing: stagnation penalty clears after movement")
        cfg = SocialSafetyRewardConfig(
            goal_progress_scale=0.0,
            checkpoint_reward_enabled=False,
            stagnation_penalty_enabled=True,
            stagnation_window_steps=12,
            stagnation_min_displacement=0.1,
            stagnation_penalty=-0.02,
        )
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot)
        reset_stagnation_tracking(env)

        for tick in range(1, 12):
            env.ticks = tick
            compute_social_safety_reward(
                env,
                previous_goal_distance=10.0,
                config=cfg,
                reached_goal=False,
                timeout=False,
                collision=False,
            )

        robot.x = 0.2
        env.ticks = 12
        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertAlmostEqual(reward, 0.0)
        self.assertFalse(info["stagnation_stalled"])
        self.assertGreaterEqual(info["stagnation_displacement"], cfg.stagnation_min_displacement)

    def test_stagnation_penalty_does_not_modify_terminal_rewards(self):
        """Testing: stagnation penalty is not applied to goal, timeout, or collision returns."""
        print("Testing: stagnation penalty leaves terminal rewards unchanged")
        cfg = SocialSafetyRewardConfig(
            stagnation_penalty_enabled=True,
            stagnation_window_steps=1,
            stagnation_min_displacement=10.0,
            stagnation_penalty=-0.02,
        )
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot)

        reward, info = compute_social_safety_reward(env, previous_goal_distance=10.0, config=cfg, reached_goal=True)
        self.assertEqual(reward, cfg.goal_reward)
        self.assertEqual(info["stagnation_penalty"], 0.0)

        reward, info = compute_social_safety_reward(env, previous_goal_distance=10.0, config=cfg, timeout=True)
        self.assertEqual(reward, cfg.timeout_penalty)
        self.assertEqual(info["stagnation_penalty"], 0.0)

        reward, info = compute_social_safety_reward(env, previous_goal_distance=10.0, config=cfg, collision=True)
        self.assertEqual(reward, cfg.object_collision_penalty)
        self.assertEqual(info["stagnation_penalty"], 0.0)

    def test_reward_uses_active_waypoint_progress_when_waypoints_are_available(self):
        """Testing: progress shaping targets the first unreached waypoint."""
        print("Testing: waypoint progress reward branch")
        cfg = SocialSafetyRewardConfig()
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(2.0, 0.0), (5.0, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["progress_target"], "waypoint")
        self.assertEqual(info["progress_target_index"], 0)
        self.assertAlmostEqual(info["progress_target_distance"], 2.0)
        self.assertAlmostEqual(reward, 0.0)

        robot.x = 0.5
        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["progress_target_index"], 0)
        self.assertAlmostEqual(info["progress_target_distance"], 1.5)
        self.assertAlmostEqual(reward, cfg.goal_progress_scale * (2.0 - 1.5))

    def test_checkpoint_reward_is_paid_once_when_robot_enters_advance_radius(self):
        """Testing: waypoint checkpoints pay once and then remain reached."""
        print("Testing: checkpoint reward pays once per waypoint")
        cfg = SocialSafetyRewardConfig(checkpoint_reward=0.5, waypoint_advance_radius=0.3)
        robot = DummyRobot(x=1.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(1.2, 0.0), (3.0, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=9.5,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 1)
        self.assertAlmostEqual(info["checkpoint_reward"], cfg.checkpoint_reward)
        self.assertAlmostEqual(reward, cfg.checkpoint_reward)

        robot.x = 1.5
        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=9.5,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 0)
        self.assertAlmostEqual(info["checkpoint_reward"], 0.0)
        self.assertEqual(info["progress_target"], "waypoint")
        self.assertEqual(info["progress_target_index"], 1)
        progress_reward = cfg.goal_progress_scale * (2.0 - 1.5)
        self.assertAlmostEqual(reward, progress_reward)

    def test_passed_waypoint_advances_without_checkpoint_reward(self):
        """Testing: passed waypoints advance progress without paying checkpoint reward."""
        print("Testing: passed waypoint advances without checkpoint payment")
        cfg = SocialSafetyRewardConfig(checkpoint_reward=0.5, waypoint_advance_radius=0.3)
        robot = DummyRobot(x=4.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(1.0, 0.0), (3.0, 0.0), (6.0, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=6.5,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 0)
        self.assertAlmostEqual(info["checkpoint_reward"], 0.0)
        self.assertEqual(info["progress_target"], "waypoint")
        self.assertEqual(info["progress_target_index"], 2)
        self.assertAlmostEqual(info["progress_target_distance"], 2.0)
        self.assertAlmostEqual(reward, 0.0)

    def test_waypoint_progress_target_advances_with_checkpoint_visits_without_jump_reward(self):
        """Testing: progress target syncs with checkpoint visits used by the sliding waypoint window."""
        print("Testing: waypoint progress target advances with checkpoint visits")
        cfg = SocialSafetyRewardConfig(checkpoint_reward=0.5, waypoint_advance_radius=0.3)
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(0.0, 0.0), (3.0, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 1)
        self.assertEqual(info["progress_target"], "waypoint")
        self.assertEqual(info["progress_target_index"], 1)
        self.assertAlmostEqual(info["progress_target_distance"], 3.0)
        self.assertAlmostEqual(reward, cfg.checkpoint_reward)

        robot.x = 1.0
        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["progress_target_index"], 1)
        self.assertAlmostEqual(info["progress_target_distance"], 2.0)
        self.assertAlmostEqual(reward, cfg.goal_progress_scale * (3.0 - 2.0))

    def test_warning_zone_and_checkpoint_rewards_are_additive(self):
        """Testing: warning-zone and checkpoint shaping are summed."""
        print("Testing: warning-zone and checkpoint rewards are additive")
        cfg = SocialSafetyRewardConfig(checkpoint_reward=0.5, waypoint_advance_radius=0.3)
        human = make_human(x=0.0, y=0.0, orientation=0.0, speed=1.0)
        robot = DummyRobot(x=0.5, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, dynamic_humans=[human], waypoints=[(0.5, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["warning_zone_hits"], 1)
        self.assertEqual(info["checkpoint_hits"], 1)
        self.assertAlmostEqual(info["checkpoint_reward"], cfg.checkpoint_reward)
        self.assertAlmostEqual(reward, info["warning_zone_reward"] + cfg.checkpoint_reward + cfg.goal_progress_scale * 0.5)

    def test_checkpoint_reached_index_resets_when_waypoint_path_changes(self):
        """Testing: checkpoint reach state belongs to the current waypoint path."""
        print("Testing: checkpoint reward reached index resets for a new waypoint path")
        cfg = SocialSafetyRewardConfig(checkpoint_reward=0.5, waypoint_advance_radius=0.3)
        robot = DummyRobot(x=1.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(1.2, 0.0)])

        compute_social_safety_reward(env, previous_goal_distance=9.5, config=cfg, reached_goal=False, timeout=False, collision=False)
        _, info = compute_social_safety_reward(env, previous_goal_distance=9.5, config=cfg, reached_goal=False, timeout=False, collision=False)
        self.assertEqual(info["checkpoint_hits"], 0)

        env.waypoints = [(1.1, 0.0)]
        _, info = compute_social_safety_reward(env, previous_goal_distance=9.5, config=cfg, reached_goal=False, timeout=False, collision=False)
        self.assertEqual(info["checkpoint_hits"], 1)

    def test_waypoint_progress_advances_when_checkpoint_reward_is_disabled(self):
        """Testing: waypoint tracking is independent of checkpoint reward payment."""
        print("Testing: waypoint progress advances even when checkpoint reward is disabled")
        cfg = SocialSafetyRewardConfig(checkpoint_reward_enabled=False, waypoint_advance_radius=0.3)
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(0.0, 0.0), (3.0, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 0)
        self.assertEqual(info["progress_target"], "waypoint")
        self.assertEqual(info["progress_target_index"], 1)
        self.assertAlmostEqual(reward, 0.0)

    def test_checkpoint_reward_uses_published_advance_radius(self):
        """Testing: checkpoint reward uses the navigation advance radius."""
        print("Testing: checkpoint reward uses shared advance radius")
        cfg = SocialSafetyRewardConfig(waypoint_advance_radius=0.0)
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(0.0, 0.0), (3.0, 0.0)])
        setattr(env, WAYPOINT_ADVANCE_RADIUS_ATTR, 0.3)

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 1)
        self.assertEqual(info["progress_target_index"], 1)
        self.assertAlmostEqual(reward, cfg.checkpoint_reward)

    def test_checkpoint_reward_uses_config_advance_radius_without_navigation_wrapper(self):
        """Testing: checkpoint reward has an explicit config fallback when wrapper state is absent."""
        print("Testing: checkpoint reward uses config fallback advance radius")
        cfg = SocialSafetyRewardConfig(waypoint_advance_radius=0.3)
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = make_env(robot, waypoints=[(0.0, 0.0), (3.0, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 1)
        self.assertEqual(info["progress_target_index"], 1)
        self.assertAlmostEqual(reward, cfg.checkpoint_reward)

    def test_waypoints_fallback_prefers_sampled_plan_waypoints_over_dense_path(self):
        """Testing: reward uses sampled A* waypoints when get_waypoints is unavailable."""
        print("Testing: waypoint fallback uses sampled plan waypoints")
        cfg = SocialSafetyRewardConfig()
        robot = DummyRobot(x=0.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = SimpleNamespace(
            robot=robot,
            ticks=0,
            EPISODE_LENGTH=100,
            GOAL_THRESHOLD=0.5,
            static_humans=[],
            dynamic_humans=[],
            plants=[],
            walls=[],
            tables=[],
            laptops=[],
            moving_interactions=[],
            static_interactions=[],
            h_l_interactions=[],
            latest_plan=SimpleNamespace(path_world=[(1.0, 0.0), (1.5, 0.0)], waypoints=[(2.0, 0.0), (5.0, 0.0)]),
        )

        _reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=10.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["progress_target_index"], 0)
        self.assertAlmostEqual(info["progress_target_distance"], 2.0)

    def test_reward_reads_waypoints_published_on_base_env(self):
        """Testing: reward can use waypoints published by an outer A* wrapper."""
        print("Testing: reward reads shared base-env waypoints")
        cfg = SocialSafetyRewardConfig(checkpoint_reward=50.0, waypoint_advance_radius=0.3)
        robot = DummyRobot(x=1.0, y=0.0, goal_x=10.0, goal_y=0.0)
        env = SimpleNamespace(
            robot=robot,
            ticks=0,
            EPISODE_LENGTH=100,
            GOAL_THRESHOLD=0.5,
            static_humans=[],
            dynamic_humans=[],
            plants=[],
            walls=[],
            tables=[],
            laptops=[],
            moving_interactions=[],
            static_interactions=[],
            h_l_interactions=[],
        )
        setattr(env, CURRENT_WAYPOINTS_ATTR, [(1.2, 0.0), (3.0, 0.0)])

        reward, info = compute_social_safety_reward(
            env,
            previous_goal_distance=9.0,
            config=cfg,
            reached_goal=False,
            timeout=False,
            collision=False,
        )

        self.assertEqual(info["reward_reason"], "shaped")
        self.assertEqual(info["checkpoint_hits"], 1)
        self.assertAlmostEqual(reward, 50.0)

    def test_socnavgym_reward_adapter_returns_custom_reward_and_info(self):
        """Testing: SocNavGym RewardAPI adapter returns the project custom reward."""
        print("Testing: SocNavGym custom reward adapter returns project reward")
        env = DummySocNavRewardEnv()
        reward_calculator = Reward(env)

        env.robot.x = 1.0
        reward = reward_calculator.compute_reward([1.0, 0.0, 0.0], {}, {})

        self.assertAlmostEqual(reward, reward_calculator.config.goal_progress_scale)
        self.assertAlmostEqual(reward_calculator.info["custom_reward"], reward)
        self.assertEqual(reward_calculator.info["reward_reason"], "shaped")
        self.assertEqual(reward_calculator.info["progress_target"], "goal")
        self.assertAlmostEqual(reward_calculator.info["distance_reward"], reward)

    def test_socnavgym_reward_adapter_uses_reward_api_terminal_checks(self):
        """Testing: SocNavGym RewardAPI adapter uses terminal helper checks."""
        print("Testing: SocNavGym custom reward adapter uses terminal checks")
        env = DummySocNavRewardEnv()
        reward_calculator = Reward(env)

        env.robot.x = 10.0
        reward = reward_calculator.compute_reward([1.0, 0.0, 0.0], {}, {})

        self.assertAlmostEqual(reward, reward_calculator.config.goal_reward)
        self.assertEqual(reward_calculator.info["reward_reason"], "goal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
