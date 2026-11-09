from __future__ import annotations

import unittest

from experiments.halora.config import load_experiment_config
from experiments.halora.model import ViTHaLoRAClassifier
from vit_core.pretrained import load_pretrained_vit


class PretrainedLoadingTest(unittest.TestCase):
    def test_full_vit_loads_into_wrapped_projections(self) -> None:
        config = load_experiment_config("configs/halora.yaml")
        model = ViTHaLoRAClassifier(
            num_classes=10,
            lora=config.lora,
            noise=config.noise,
        )
        loaded = load_pretrained_vit(model)
        self.assertGreater(loaded, 0)
        self.assertEqual(len(model.lora_modules()), 60)
        self.assertEqual(len(model.noisy_modules()), 72)
        lora_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".lora_down." in name or ".lora_up." in name
        )
        self.assertEqual(lora_parameters, 589_824)


if __name__ == "__main__":
    unittest.main()
