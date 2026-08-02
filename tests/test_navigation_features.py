import math
import unittest
from types import SimpleNamespace

import gym
import numpy as np
from gym import spaces

from navigation_features import CoordinateFrameWaypointWrapper, NearestWallSegmentWrapper
from navigation_features.waypoint_state import (
    LAST_REACHED_WAYPOINT_ATTR,
    PROGRESS_TARGET_DISTANCE_ATTR,
    PROGRESS_TARGET_SIGNATURE_ATTR,
    advance_waypoint_index,
)


class DummyPlanningEnv(gym.Env):
    """Minimal env for testing coordinate-frame and waypoint features without SocNavGym."""

    MAP_X = 20.0
    MAP_Y = 20.0
    PIXEL_TO_WORLD_X = 10.0
    PIXEL_TO_WORLD_Y = 10.0

    def __init__(self):
        super().__init__()
        self.robot = SimpleNamespace(x=0.0, y=0.0, orientation=0.0, goal_x=3.0, goal_y=4.0)
        self.latest_plan = object()
        self.render_callbacks = []
        self.observation_space = spaces.Dict(
            {
                "robot": spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
                "humans": spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    @property
    def unwrapped(self):
        return self

    def reset(self, **kwargs):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}

    def get_waypoints(self, interval=None):
        return [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]

    def _obs(self):
        robot_obs = np.array([1, 0, 0, 0, 0, 0, 3.0, 4.0, 0.25], dtype=np.float32)
        human_obs = np.array([0, 1, 0, 0, 0, 0, 1.0, 0.0, 0.0, 1.0, 0.36, 0.0, 0.0, 0.0], dtype=np.float32)
        return {"robot": robot_obs, "humans": human_obs}


class IntermittentPlanningEnv(DummyPlanningEnv):
    """Dummy env whose first reset has no waypoints, then later resets succeed."""

    def __init__(self, waypoint_sequences):
        super().__init__()
        self.waypoint_sequences = list(waypoint_sequences)
        self.reset_calls = 0
        self.reset_seeds = []

    def reset(self, **kwargs):
        self.reset_calls += 1
        self.reset_seeds.append(kwargs.get("seed"))
        return self._obs(), {}

    def get_waypoints(self, interval=None):
        index = min(self.reset_calls - 1, len(self.waypoint_sequences) - 1)
        return list(self.waypoint_sequences[index])


class PlannerErrorEnv(DummyPlanningEnv):
    """Dummy env that raises an unexpected planner error when waypoints are queried."""

    def get_waypoints(self, interval=None):
        raise ValueError("planner config bug")


class DummyWallEnv(DummyPlanningEnv):
    """Dummy env with SocNavGym-like wall segment observations."""

    def __init__(self):
        super().__init__()
        self.robot.radius = 0.25
        self.walls = [SimpleNamespace(id=1, thickness=0.2), SimpleNamespace(id=2, thickness=0.2)]

    def _get_entity_obs(self, wall):
        rows = {
            1: np.array(
                [
                    [0, 0, 0, 0, 0, 1, 5.0, 0.0, 0, 1, 0.5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 1.0, 0.0, 0, 1, 0.5, 0, 0, 0],
                ],
                dtype=np.float32,
            ),
            2: np.array([[0, 0, 0, 0, 0, 1, 3.0, 0.0, 0, 1, 0.5, 0, 0, 0]], dtype=np.float32),
        }
        return rows[wall.id].reshape(-1)


class SurfaceRankingWallEnv(DummyWallEnv):
    """Wall rows whose centre and surface distance rankings disagree."""

    def _get_entity_obs(self, wall):
        rows = {
            # Its centre is farther away, but the long segment ends near the robot.
            1: np.array([[0, 0, 0, 0, 0, 1, 2.0, 0.0, 0, 1, 1.5, 0, 0, 0]], dtype=np.float32),
            2: np.array([[0, 0, 0, 0, 0, 1, 0.0, 1.0, 0, 1, 0.5, 0, 0, 0]], dtype=np.float32),
        }
        return rows[wall.id].reshape(-1)


class BoundaryFilteringWallEnv(DummyPlanningEnv):
    """Square room with two perimeter walls and two internal corridor walls."""

    def __init__(self):
        super().__init__()
        self.shape = "square"
        self.MAP_X = 10.0
        self.MAP_Y = 10.0
        self.robot.radius = 0.25
        self.walls = [
            SimpleNamespace(id=1, x=0.0, y=4.9, orientation=0.0, length=10.0, thickness=0.2),
            SimpleNamespace(id=2, x=-4.9, y=0.0, orientation=math.pi / 2, length=10.0, thickness=0.2),
            SimpleNamespace(id=3, x=-3.0, y=-1.7, orientation=0.0, length=4.0, thickness=0.2),
            SimpleNamespace(id=4, x=3.0, y=1.7, orientation=0.0, length=4.0, thickness=0.2),
        ]

    def _get_entity_obs(self, wall):
        row = np.zeros((1, 14), dtype=np.float32)
        row[0, 5] = 1.0
        row[0, 6] = float(wall.id)
        row[0, 9] = 1.0
        row[0, 10] = float(wall.length) / 2.0
        return row.reshape(-1)


class TestNavigationFeatures(unittest.TestCase):
    """Tests for coordinate-frame transforms and waypoint feature generation."""

    def make_wrapper(
        self,
        mode="heading_aligned",
        max_waypoints=3,
        flatten=True,
        advance_radius=0.3,
        visualization_enabled=False,
    ):
        """Create wrapper with deterministic waypoint settings for exact assertions."""
        print(f"Testing setup: navigation wrapper mode={mode}, max_waypoints={max_waypoints}, flatten={flatten}")
        return CoordinateFrameWaypointWrapper(
            DummyPlanningEnv(),
            config={
                "coordinate_frame": {
                    "mode": mode,
                    "transform_observations": True,
                },
                "waypoint_features": {
                    "max_waypoints": max_waypoints,
                    "flatten": flatten,
                    "advance_radius": advance_radius,
                    "replan_if_missing": False,
                },
                "visualization": {
                    "enabled": visualization_enabled,
                    "draw_slot_labels": False,
                },
            },
        )

    def make_wrapper_for_env(self, env, max_reset_attempts=20):
        """Create wrapper around a custom dummy env for reset failure tests."""
        return CoordinateFrameWaypointWrapper(
            env,
            config={
                "waypoint_features": {
                    "max_waypoints": 2,
                    "flatten": False,
                    "advance_radius": 0.3,
                    "replan_if_missing": False,
                    "max_reset_attempts": max_reset_attempts,
                },
            },
        )

    def test_heading_aligned_frame_leaves_original_observations_unchanged(self):
        """Testing: heading-aligned mode preserves SocNavGym's default observation frame."""
        print("Testing: heading-aligned observations remain unchanged")
        wrapper = self.make_wrapper(mode="heading_aligned")
        raw_obs = wrapper.env._obs()

        obs, _info = wrapper.reset()

        np.testing.assert_allclose(obs["robot"], raw_obs["robot"])

    def test_nearest_wall_segments_are_fixed_size_and_padded(self):
        """Testing: nearest wall wrapper exposes nearest rows and pads missing slots."""
        print("Testing: nearest wall segment observations are fixed-size and padded")
        wrapper = NearestWallSegmentWrapper(DummyWallEnv(), count=4)

        obs, _info = wrapper.reset()
        walls = obs["walls"].reshape(4, 14)

        self.assertEqual(wrapper.observation_space.spaces["walls"].shape, (56,))
        self.assertEqual(obs["walls"].shape, (56,))
        self.assertAlmostEqual(walls[0, 6], 1.0)
        self.assertAlmostEqual(walls[1, 6], 3.0)
        self.assertAlmostEqual(walls[2, 6], 5.0)
        np.testing.assert_allclose(walls[3], np.zeros(14, dtype=np.float32))

    def test_nearest_wall_segments_are_ranked_by_surface_clearance(self):
        """Testing: finite wall surfaces, rather than segment centres, determine ranking."""
        wrapper = NearestWallSegmentWrapper(SurfaceRankingWallEnv(), count=2)

        obs, _info = wrapper.reset()
        walls = obs["walls"].reshape(2, 14)

        self.assertAlmostEqual(walls[0, 6], 2.0)
        self.assertAlmostEqual(walls[0, 7], 0.0)
        self.assertAlmostEqual(walls[1, 6], 0.0)
        self.assertAlmostEqual(walls[1, 7], 1.0)

    def test_all_wall_segments_preserve_socnavgym_order_and_pad(self):
        """Testing: all mode preserves source order instead of distance ranking."""
        wrapper = NearestWallSegmentWrapper(DummyWallEnv(), count=4, mode="all")

        obs, _info = wrapper.reset()
        walls = obs["walls"].reshape(4, 14)

        np.testing.assert_array_equal(walls[:3, 6], [5.0, 1.0, 3.0])
        np.testing.assert_array_equal(walls[3], np.zeros(14, dtype=np.float32))

    def test_all_wall_segments_reject_capacity_overflow(self):
        """Testing: all mode never silently discards wall segments."""
        wrapper = NearestWallSegmentWrapper(DummyWallEnv(), count=2, mode="all")

        with self.assertRaisesRegex(ValueError, "produced 3 segments"):
            wrapper.reset()

    def test_boundary_filter_keeps_corridors_and_pads_remaining_capacity(self):
        """Testing: geometry filtering removes perimeter walls before dynamic padding."""
        wrapper = NearestWallSegmentWrapper(
            BoundaryFilteringWallEnv(),
            count=3,
            mode="all",
            include_boundary_walls=False,
        )

        obs, _info = wrapper.reset()
        walls = obs["walls"].reshape(3, 14)

        np.testing.assert_array_equal(walls[:2, 6], [3.0, 4.0])
        np.testing.assert_array_equal(walls[2], np.zeros(14, dtype=np.float32))

    def test_wall_segment_mode_is_validated(self):
        """Testing: unsupported wall selection modes fail at construction."""
        with self.assertRaisesRegex(ValueError, '"nearest" or "all"'):
            NearestWallSegmentWrapper(DummyWallEnv(), count=4, mode="unknown")

    def test_goal_aligned_frame_rotates_robot_goal_to_positive_x_axis(self):
        """Testing: goal-aligned mode rotates robot goal to [distance, 0]."""
        print("Testing: goal-aligned robot goal becomes [distance, 0]")
        wrapper = self.make_wrapper(mode="goal_aligned")

        obs, _info = wrapper.reset()

        np.testing.assert_allclose(obs["robot"][6:8], np.array([5.0, 0.0], dtype=np.float32), atol=1e-6)

    def test_waypoint_feature_vector_has_expected_flat_shape(self):
        """Testing: flat waypoint feature vector has max_waypoints * feature_dim values."""
        print("Testing: waypoint feature vector flat shape")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=3, flatten=True)

        obs, _info = wrapper.reset()

        self.assertIn("waypoint_features", obs)
        self.assertEqual(obs["waypoint_features"].shape, (12,))
        self.assertEqual(wrapper.observation_space["waypoint_features"].shape, (12,))

    def test_waypoint_feature_matrix_has_expected_sequence_shape_when_not_flattened(self):
        """Testing: non-flat waypoint features keep [max_waypoints, feature_dim] sequence shape."""
        print("Testing: waypoint feature matrix sequence shape")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=3, flatten=False)

        obs, _info = wrapper.reset()

        self.assertEqual(obs["waypoint_features"].shape, (3, 4))
        self.assertEqual(wrapper.observation_space["waypoint_features"].shape, (3, 4))

    def test_goal_aligned_waypoint_on_goal_direction_has_zero_bearing(self):
        """Testing: waypoint lying on the robot-goal direction has zero bearing in goal-aligned frame."""
        print("Testing: goal-aligned waypoint bearing is zero for goal-direction waypoint")
        wrapper = self.make_wrapper(mode="goal_aligned", max_waypoints=3, flatten=False)

        obs, _info = wrapper.reset()
        second_waypoint = obs["waypoint_features"][1]

        np.testing.assert_allclose(second_waypoint[:2], np.array([5.0, 0.0], dtype=np.float32), atol=1e-6)
        self.assertAlmostEqual(float(second_waypoint[2]), 5.0, places=6)
        self.assertAlmostEqual(float(second_waypoint[3]), 0.0, places=6)

    def test_waypoint_window_advances_after_reaching_checkpoint_radius(self):
        """Testing: waypoint feature window advances after the current waypoint is reached."""
        print("Testing: waypoint feature window slides after reaching a waypoint")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False)

        initial_obs, _info = wrapper.reset()
        np.testing.assert_allclose(initial_obs["waypoint_features"][0, :2], np.array([0.0, 0.0], dtype=np.float32))

        next_obs, *_ = wrapper.step(np.zeros(1, dtype=np.float32))

        np.testing.assert_allclose(next_obs["waypoint_features"][0, :2], np.array([3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(next_obs["waypoint_features"][0, 2]), 5.0, places=6)

    def test_waypoint_window_advances_after_passing_checkpoint(self):
        """Testing: waypoint feature window can skip a passed waypoint without a radius hit."""
        print("Testing: waypoint feature window slides after passing a waypoint")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False)
        wrapper.reset()
        wrapper.step(np.zeros(1, dtype=np.float32))
        wrapper.env.robot.x = 4.0
        wrapper.env.robot.y = 5.0

        next_obs, *_ = wrapper.step(np.zeros(1, dtype=np.float32))

        np.testing.assert_allclose(next_obs["waypoint_features"][0, :2], np.array([2.0, 3.0], dtype=np.float32))
        self.assertEqual(getattr(wrapper.unwrapped, LAST_REACHED_WAYPOINT_ATTR), 1)

    def test_passed_waypoint_check_handles_diagonal_path(self):
        """Testing: passed-waypoint projection works for non-axis-aligned paths."""
        print("Testing: passed waypoint projection works on a diagonal path")
        waypoints = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]

        last_reached, hits = advance_waypoint_index(
            robot_x=3.3,
            robot_y=4.4,
            waypoints=waypoints,
            last_reached_index=0,
            advance_radius=0.3,
        )

        self.assertEqual(last_reached, 1)
        self.assertEqual(hits, 0)

    def test_repeated_waypoint_feature_reads_do_not_advance_window(self):
        """Testing: feature construction is read-only and does not mutate waypoint progress."""
        print("Testing: repeated waypoint feature reads keep the same window")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False)
        wrapper.reset()

        first_read = wrapper.get_waypoint_features()
        second_read = wrapper.get_waypoint_features()

        np.testing.assert_allclose(first_read, second_read)
        np.testing.assert_allclose(second_read[0, :2], np.array([0.0, 0.0], dtype=np.float32))

    def test_waypoint_window_syncs_with_shared_reward_progress_index(self):
        """Testing: waypoint feature window honors the shared reward-side active index."""
        print("Testing: waypoint window syncs with shared reward progress index")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False)
        wrapper.reset()
        setattr(wrapper.unwrapped, LAST_REACHED_WAYPOINT_ATTR, 0)

        features = wrapper.get_waypoint_features()

        np.testing.assert_allclose(features[0, :2], np.array([3.0, 4.0], dtype=np.float32))
        self.assertEqual(wrapper._last_reached_waypoint_index, 0)

    def test_waypoint_reset_clears_shared_progress_distance_cache(self):
        """Testing: waypoint reset clears stale reward-side progress target distances."""
        print("Testing: waypoint reset clears shared progress distance cache")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False)
        setattr(wrapper.unwrapped, PROGRESS_TARGET_SIGNATURE_ATTR, ("waypoint", 0, 1.0, 0.0))
        setattr(wrapper.unwrapped, PROGRESS_TARGET_DISTANCE_ATTR, 1.0)

        wrapper.reset()

        self.assertFalse(hasattr(wrapper.unwrapped, PROGRESS_TARGET_SIGNATURE_ATTR))
        self.assertFalse(hasattr(wrapper.unwrapped, PROGRESS_TARGET_DISTANCE_ATTR))

    def test_waypoint_advance_radius_comes_from_navigation_config(self):
        """Testing: render checkpoint radius does not control waypoint feature progression."""
        print("Testing: waypoint feature window uses navigation advance radius, not render radius")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False, advance_radius=0.3)
        wrapper.env.config = {"render": {"checkpoint_radius": 0.0}}

        wrapper.reset()
        next_obs, *_ = wrapper.step(np.zeros(1, dtype=np.float32))

        np.testing.assert_allclose(next_obs["waypoint_features"][0, :2], np.array([3.0, 4.0], dtype=np.float32))
        self.assertEqual(getattr(wrapper.unwrapped, LAST_REACHED_WAYPOINT_ATTR), 0)

    def test_short_waypoint_window_duplicates_last_available_waypoint(self):
        """Testing: missing waypoint slots repeat the last available waypoint instead of zero padding."""
        print("Testing: short waypoint windows duplicate the last available waypoint")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False)

        wrapper.reset()
        wrapper.step(np.zeros(1, dtype=np.float32))
        wrapper.env.robot.x = 3.0
        wrapper.env.robot.y = 4.0
        obs, *_ = wrapper.step(np.zeros(1, dtype=np.float32))

        np.testing.assert_allclose(obs["waypoint_features"][0, :2], np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(obs["waypoint_features"][1, :2], np.array([3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(obs["waypoint_features"][0], obs["waypoint_features"][1])

    def test_waypoint_visualization_uses_same_window_as_observation_features(self):
        """Testing: render overlay draws the current waypoint slots exposed to the policy."""
        print("Testing: waypoint visualization follows the observation window")
        wrapper = self.make_wrapper(mode="heading_aligned", max_waypoints=2, flatten=False, visualization_enabled=True)
        wrapper.reset()
        wrapper.step(np.zeros(1, dtype=np.float32))
        wrapper.env.robot.x = 3.0
        wrapper.env.robot.y = 4.0
        obs, *_ = wrapper.step(np.zeros(1, dtype=np.float32))
        image = np.zeros((200, 200, 3), dtype=np.uint8)

        wrapper.draw_waypoint_window_overlay(image, wrapper.unwrapped)

        self.assertEqual(len(wrapper.unwrapped.render_callbacks), 1)
        np.testing.assert_allclose(obs["waypoint_features"][0, :2], np.array([3.0, 4.0], dtype=np.float32))
        pixel = wrapper._world_to_pixel((6.0, 8.0))
        np.testing.assert_array_equal(image[pixel[1], pixel[0] + 8], np.array([0, 0, 255], dtype=np.uint8))
        np.testing.assert_array_equal(image[pixel[1], pixel[0] + 10], np.array([255, 255, 0], dtype=np.uint8))
        skipped_pixel = wrapper._world_to_pixel((3.0, 4.0))
        np.testing.assert_array_equal(image[skipped_pixel[1], skipped_pixel[0] + 4], np.array([120, 120, 120], dtype=np.uint8))

    def test_unseeded_reset_skips_episode_when_planner_produces_no_waypoints(self):
        """Testing: reset retries internally until waypoint guidance exists."""
        print("Testing: reset skips planner-failed episodes before exposing observation")
        env = IntermittentPlanningEnv([[], [(1.0, 0.0), (2.0, 0.0)]])
        wrapper = self.make_wrapper_for_env(env)

        obs, info = wrapper.reset()

        self.assertEqual(env.reset_calls, 2)
        self.assertEqual(env.reset_seeds, [None, None])
        self.assertEqual(info["planner_failed_reset_attempts"], 1)
        np.testing.assert_allclose(obs["waypoint_features"][0, :2], np.array([1.0, 0.0], dtype=np.float32))

    def test_seeded_reset_fails_fast_when_planner_produces_no_waypoints(self):
        """Testing: fixed seeded reset does not retry the same no-waypoint scenario."""
        print("Testing: seeded planner failure fails fast")
        env = IntermittentPlanningEnv([[], [(1.0, 0.0), (2.0, 0.0)]])
        wrapper = self.make_wrapper_for_env(env)

        with self.assertRaisesRegex(RuntimeError, "Planner produced no waypoints"):
            wrapper.reset(seed=10)

        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.reset_seeds, [10])

    def test_reset_raises_when_planner_never_produces_waypoints(self):
        """Testing: repeated planner failure has a bounded retry count."""
        print("Testing: reset raises after repeated planner failures")
        env = IntermittentPlanningEnv([[], [], []])
        wrapper = self.make_wrapper_for_env(env, max_reset_attempts=3)

        with self.assertRaisesRegex(RuntimeError, "Planner produced no waypoints"):
            wrapper.reset()

        self.assertEqual(env.reset_calls, 3)

    def test_reset_does_not_swallow_unexpected_planner_errors(self):
        """Testing: unexpected planner errors are raised instead of retried as empty plans."""
        print("Testing: reset propagates unexpected planner errors")
        wrapper = self.make_wrapper_for_env(PlannerErrorEnv(), max_reset_attempts=3)

        with self.assertRaisesRegex(ValueError, "planner config bug"):
            wrapper.reset()


if __name__ == "__main__":
    unittest.main(verbosity=2)
