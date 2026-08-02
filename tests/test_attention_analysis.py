import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from environment_inspection.attention_analysis import (
    _normalized_entropy,
    _spearman,
    extract_entity_rows,
    resolve_analysis_cases,
    resolve_analysis_seeds,
)


class _Robot:
    x = 2.0
    y = -1.0
    orientation = np.pi / 2


class _Env:
    robot = _Robot()


class AttentionAnalysisTests(unittest.TestCase):
    def test_entity_rows_match_policy_order_mask_and_coordinate_transform(self):
        human = np.zeros((2, 14), dtype=np.float32)
        human[0, 0] = 1.0
        human[0, 6:8] = [1.0, 0.0]
        human[0, 10] = 0.3
        observation = {
            "robot": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.25], dtype=np.float32),
            "humans": human.reshape(-1),
        }

        rows = extract_entity_rows(observation, ["humans"], 14, _Env())

        self.assertEqual([row["label"] for row in rows], ["E01", "E02"])
        self.assertTrue(rows[0]["valid"])
        self.assertFalse(rows[1]["valid"])
        self.assertAlmostEqual(rows[0]["world_x"], 2.0)
        self.assertAlmostEqual(rows[0]["world_y"], 0.0)
        self.assertAlmostEqual(rows[0]["clearance"], 0.45)

    def test_seed_selection_is_sorted_per_category_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            with open(path, "w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["seed", "agent_SUCCESS", "orca_SUCCESS", "agent_COLLISION", "agent_TIMEOUT"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"seed": 12, "agent_SUCCESS": 1, "orca_SUCCESS": 0, "agent_COLLISION": 0, "agent_TIMEOUT": 0},
                        {"seed": 10, "agent_SUCCESS": 1, "orca_SUCCESS": 0, "agent_COLLISION": 0, "agent_TIMEOUT": 0},
                        {"seed": 11, "agent_SUCCESS": 0, "orca_SUCCESS": 1, "agent_COLLISION": 1, "agent_TIMEOUT": 0},
                    ]
                )
            settings = {
                "seeds": [],
                "selection": {
                    "comparison_csv": str(path),
                    "per_category": 1,
                    "categories": ["agent_success_orca_failure", "agent_collision"],
                },
            }

            self.assertEqual(resolve_analysis_seeds(settings), [10, 11])
            self.assertEqual(
                resolve_analysis_cases(settings),
                [
                    {"seed": 10, "category": "agent_success_orca_failure"},
                    {"seed": 11, "category": "agent_collision"},
                ],
            )

    def test_attention_statistics_have_expected_limits(self):
        self.assertAlmostEqual(_normalized_entropy(np.array([0.5, 0.5])), 1.0)
        self.assertAlmostEqual(_normalized_entropy(np.array([1.0, 0.0])), 0.0)
        self.assertAlmostEqual(_spearman([1, 2, 3], [3, 2, 1]), -1.0)

    def test_random_seed_selection_is_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            with open(path, "w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["seed", "agent_SUCCESS", "orca_SUCCESS", "agent_COLLISION", "agent_TIMEOUT"],
                )
                writer.writeheader()
                for seed in range(10, 20):
                    writer.writerow(
                        {
                            "seed": seed,
                            "agent_SUCCESS": 1,
                            "orca_SUCCESS": 0,
                            "agent_COLLISION": 0,
                            "agent_TIMEOUT": 0,
                        }
                    )
            settings = {
                "seeds": [],
                "selection": {
                    "comparison_csv": str(path),
                    "mode": "random",
                    "random_seed": 3042,
                    "per_category": 3,
                    "categories": ["agent_success_orca_failure"],
                },
            }

            first = resolve_analysis_cases(settings)
            second = resolve_analysis_cases(settings)

            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertNotEqual([case["seed"] for case in first], [10, 11, 12])


if __name__ == "__main__":
    unittest.main()
