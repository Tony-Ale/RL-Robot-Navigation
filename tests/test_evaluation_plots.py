import csv
from pathlib import Path
import tempfile
import unittest

from testing_pipeline.evaluation_plots import (
    generate_evaluation_plots,
    load_architecture_series,
    load_final_model_rows,
    load_training_series,
)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class TestEvaluationPlots(unittest.TestCase):
    def test_architecture_comparison_uses_only_common_steps_within_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.csv"
            second = Path(tmpdir) / "second.csv"
            write_csv(
                first,
                [
                    {"checkpoint_step": 50, "controller": "learned_agent", "success_rate": 0.2},
                    {"checkpoint_step": 100, "controller": "learned_agent", "success_rate": 0.4},
                    {"checkpoint_step": 150, "controller": "learned_agent", "success_rate": 0.6},
                ],
            )
            write_csv(
                second,
                [
                    {"checkpoint_step": 50, "controller": "learned_agent", "success_rate": 0.3},
                    {"checkpoint_step": 100, "controller": "learned_agent", "success_rate": 0.5},
                    {"checkpoint_step": 200, "controller": "learned_agent", "success_rate": 0.7},
                ],
            )

            series, common_steps = load_architecture_series(
                {
                    "max_step": 150,
                    "models": [
                        {"label": "A", "summary_csv": str(first)},
                        {"label": "B", "summary_csv": str(second)},
                    ],
                }
            )

            self.assertEqual(common_steps, {50, 100})
            self.assertEqual(set(series), {"A", "B"})

    def test_final_outcomes_reject_mismatched_seed_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            episodes = Path(tmpdir) / "episodes.csv"
            write_csv(
                episodes,
                [
                    {"checkpoint_step": 100, "controller": "learned_agent", "seed": 1},
                    {"checkpoint_step": 100, "controller": "learned_agent", "seed": 2},
                    {"checkpoint_step": 100, "controller": "orca", "seed": 1},
                    {"checkpoint_step": 100, "controller": "orca", "seed": 3},
                ],
            )

            with self.assertRaisesRegex(ValueError, "identical seed sets"):
                load_final_model_rows(
                    {
                        "require_matching_seeds": True,
                        "models": [
                            {
                                "label": "Agent",
                                "episode_csv": str(episodes),
                                "checkpoint_step": 100,
                                "controller": "learned_agent",
                            },
                            {
                                "label": "ORCA",
                                "episode_csv": str(episodes),
                                "checkpoint_step": 100,
                                "controller": "orca",
                            },
                        ],
                    }
                )

    def test_training_segments_join_in_global_step_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent.csv"
            child = Path(tmpdir) / "child.csv"
            write_csv(
                parent,
                [
                    {"global_step": 10, "episode_reward": 1.0},
                    {"global_step": 20, "episode_reward": 2.0},
                    {"global_step": 30, "episode_reward": 3.0},
                ],
            )
            write_csv(
                child,
                [
                    {"global_step": 31, "episode_reward": 4.0},
                    {"global_step": 40, "episode_reward": 5.0},
                ],
            )

            series = load_training_series(
                {
                    "models": [
                        {
                            "label": "Resumed",
                            "segments": [
                                {"csv": str(parent), "end_step": 30},
                                {"csv": str(child), "start_step": 31},
                            ],
                        }
                    ]
                }
            )

            self.assertEqual([int(row["global_step"]) for row in series["Resumed"]], [10, 20, 30, 31, 40])

    def test_training_segments_reject_overlapping_branches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent = Path(tmpdir) / "parent.csv"
            child = Path(tmpdir) / "child.csv"
            write_csv(parent, [{"global_step": 10}, {"global_step": 30}])
            write_csv(child, [{"global_step": 20}, {"global_step": 40}])

            with self.assertRaisesRegex(ValueError, "overlap"):
                load_training_series(
                    {
                        "models": [
                            {
                                "label": "Invalid",
                                "segments": [{"csv": str(parent)}, {"csv": str(child)}],
                            }
                        ]
                    }
                )

    def test_three_sections_generate_separate_plot_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "summary.csv"
            episodes = root / "episodes.csv"
            training = root / "training.csv"
            write_csv(
                summary,
                [
                    {
                        "checkpoint_step": 50,
                        "controller": "learned_agent",
                        "success_rate": 0.5,
                    }
                ],
            )
            write_csv(
                episodes,
                [
                    {
                        "checkpoint_step": 50,
                        "controller": "learned_agent",
                        "seed": 1,
                        "SUCCESS": True,
                        "COLLISION": False,
                        "TIMEOUT": False,
                    }
                ],
            )
            write_csv(
                training,
                [
                    {"global_step": 10, "episode_reward": 1.0},
                    {"global_step": 20, "episode_reward": 3.0},
                ],
            )
            config = {
                "save": True,
                "show": False,
                "output_dir": "plots",
                "architecture_comparison": {
                    "enabled": True,
                    "metrics": ["success_rate"],
                    "models": [{"label": "Agent", "summary_csv": str(summary)}],
                },
                "final_outcomes": {
                    "enabled": True,
                    "models": [
                        {
                            "label": "Agent",
                            "episode_csv": str(episodes),
                            "checkpoint_step": 50,
                        }
                    ],
                    "metrics": [],
                },
                "training_curves": {
                    "enabled": True,
                    "rolling_window_episodes": 2,
                    "metrics": ["episode_reward"],
                    "models": [
                        {
                            "label": "Agent",
                            "segments": [{"csv": str(training)}],
                        }
                    ],
                },
            }

            written = generate_evaluation_plots(config, root)

            self.assertEqual(len(written), 3)
            self.assertTrue(all(path.exists() for path in written))
            self.assertEqual(
                {path.parent.name for path in written},
                {"architecture_comparison", "final_outcomes", "training_curves"},
            )


if __name__ == "__main__":
    unittest.main()
