import hashlib
from pathlib import Path
import unittest

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PRETRAINED_ROOT = REPOSITORY_ROOT / "pretrained_agents"
EXPECTED_AGENTS = {
    "feedforward_humans": "ppo",
    "fixed_window_bigru_humans": "ppo",
    "persistent_state_humans": "stateful_ppo",
    "persistent_state_humans_plants": "stateful_ppo",
}
EXPECTED_FILES = {
    "architecture_config.yaml",
    "environment_config.yaml",
    "metadata.yaml",
    "model.zip",
    "render_config.yaml",
    "reward_config.yaml",
    "training_config.yaml",
}


def _read_yaml(path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestPretrainedAgents(unittest.TestCase):
    def test_packages_are_complete_and_internally_consistent(self):
        package_names = {path.name for path in PRETRAINED_ROOT.iterdir() if path.is_dir()}
        self.assertEqual(package_names, set(EXPECTED_AGENTS))

        for name, policy_type in EXPECTED_AGENTS.items():
            with self.subTest(agent=name):
                package = PRETRAINED_ROOT / name
                self.assertEqual({path.name for path in package.iterdir()}, EXPECTED_FILES)

                metadata = _read_yaml(package / "metadata.yaml")
                render = _read_yaml(package / "render_config.yaml")["rendering"]
                training = _read_yaml(package / "training_config.yaml")

                self.assertEqual(metadata["checkpoint_sha256"], _sha256(package / "model.zip"))
                self.assertEqual(metadata["policy_type"], policy_type)
                self.assertEqual(render["policy_type"], policy_type)
                self.assertEqual(training["testing"]["policy_type"], policy_type)
                self.assertEqual(metadata["entity_keys"], training["architecture"]["entity_keys"])

                expected_prefix = f"pretrained_agents/{name}"
                self.assertEqual(render["run_dir"], expected_prefix)
                self.assertEqual(render["checkpoint_path"], f"{expected_prefix}/model.zip")
                self.assertEqual(
                    render["training_config_path"],
                    f"{expected_prefix}/training_config.yaml",
                )
                self.assertEqual(
                    training["environment"]["config_path"],
                    f"{expected_prefix}/environment_config.yaml",
                )
                self.assertEqual(
                    training["architecture"]["config_path"],
                    f"{expected_prefix}/architecture_config.yaml",
                )
                self.assertIsNone(training["training"]["resume_from_checkpoint"])


if __name__ == "__main__":
    unittest.main()
