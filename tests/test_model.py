from __future__ import annotations

import unittest

import torch

from experiments.halora.config import LoRAConfig, NoiseConfig
from experiments.halora.model import ViTHaLoRAClassifier
from vit_core.vit_backbone import ViTConfig


def tiny_model() -> ViTHaLoRAClassifier:
    return ViTHaLoRAClassifier(
        num_classes=3,
        lora=LoRAConfig(
            rank=4,
            alpha=4.0,
            targets=("query", "key", "value", "fc1", "fc2"),
        ),
        noise=NoiseConfig(
            sigma=0.01,
            block_size=8,
            refresh_steps=400,
            train_seed=1042,
            granularity="elementwise_iid",
            scope="encoder_linear_weights",
        ),
        vit_config=ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_heads=4,
            intermediate_size=64,
            num_layers=2,
        ),
    )


class HaLoRAModelTest(unittest.TestCase):
    def test_module_counts_and_trainable_scope(self) -> None:
        model = tiny_model()
        self.assertEqual(len(model.lora_modules()), 10)
        self.assertEqual(len(model.noisy_modules()), 12)
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(
                ".lora_down." in name
                or ".lora_up." in name
                or name.startswith("head.classifier.")
                for name in trainable_names
            )
        )
        lora_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".lora_down." in name or ".lora_up." in name
        )
        self.assertEqual(lora_parameters, 3072)

    def test_fixed_noise_changes_output_and_can_be_disabled(self) -> None:
        torch.manual_seed(13)
        model = tiny_model().eval()
        images = torch.randn(2, 3, 32, 32)
        model.disable_noise()
        clean = model(images)
        model.resample_noise_from_seed(1)
        model.enable_noise()
        noisy = model(images)
        self.assertFalse(torch.equal(clean, noisy))
        model.disable_noise()
        restored = model(images)
        torch.testing.assert_close(restored, clean, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
