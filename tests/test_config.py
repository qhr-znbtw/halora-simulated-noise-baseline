from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path

from experiments.halora.config import EXPECTED_TARGETS, load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentConfigTest(unittest.TestCase):
    def test_four_conditions_differ_only_in_name_and_two_flags(self) -> None:
        names = (
            "vanilla_lora",
            "noise_aware_lora",
            "regularized_lora",
            "halora",
        )
        configs = [
            load_experiment_config(PROJECT_ROOT / "configs" / f"{name}.yaml")
            for name in names
        ]
        expected_flags = {
            "vanilla_lora": (False, False),
            "noise_aware_lora": (True, False),
            "regularized_lora": (False, True),
            "halora": (True, True),
        }
        for config in configs:
            self.assertEqual(config.lora.targets, EXPECTED_TARGETS)
            self.assertEqual(
                config.regularizer.interval,
                config.noise.refresh_steps,
            )
            self.assertEqual(
                (
                    config.condition.training_noise,
                    config.condition.regularizer,
                ),
                expected_flags[config.condition.name],
            )

        reference = asdict(configs[0])
        reference.pop("condition")
        for config in configs[1:]:
            candidate = asdict(config)
            candidate.pop("condition")
            self.assertEqual(candidate, reference)


if __name__ == "__main__":
    unittest.main()
