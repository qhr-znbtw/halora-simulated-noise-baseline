from __future__ import annotations

import unittest

import torch

from experiments.halora.layers import HaLoRALinear, NoisyLinear


class HaLoRALayerTest(unittest.TestCase):
    def test_initial_lora_branch_is_exactly_zero(self) -> None:
        torch.manual_seed(3)
        layer = HaLoRALinear(
            7,
            5,
            rank=2,
            alpha=2.0,
            sigma=0.01,
            block_size=4,
        )
        inputs = torch.randn(4, 7)
        expected = layer.base(inputs)
        actual = layer(inputs)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_noise_is_reproducible_non_accumulating_and_preserves_base(self) -> None:
        torch.manual_seed(5)
        layer = NoisyLinear(7, 5, sigma=0.01, block_size=4)
        clean_weight = layer.base.weight.detach().clone()

        generator = torch.Generator(device="cpu").manual_seed(11)
        layer.resample_noise(generator)
        first_noise = layer._rram_noise.detach().clone()
        layer._rram_noise.add_(123.0)
        generator = torch.Generator(device="cpu").manual_seed(11)
        layer.resample_noise(generator)

        torch.testing.assert_close(layer._rram_noise, first_noise)
        torch.testing.assert_close(layer.base.weight, clean_weight)
        self.assertGreater(float(first_noise.abs().sum()), 0.0)

    def test_efficient_regularizer_matches_direct_large_grams(self) -> None:
        torch.manual_seed(7)
        layer = HaLoRALinear(
            9,
            6,
            rank=3,
            alpha=3.0,
            sigma=0.01,
            block_size=4,
        )
        with torch.no_grad():
            layer.lora_up.weight.normal_()
        down = layer.lora_down.weight.float()
        up = layer.lora_up.weight.float()
        direct = torch.linalg.matrix_norm(
            down.transpose(0, 1) @ down, ord="fro"
        ) + torch.linalg.matrix_norm(up @ up.transpose(0, 1), ord="fro")
        efficient = layer.regularization_loss()
        torch.testing.assert_close(efficient, direct, rtol=1e-5, atol=1e-6)

        efficient.backward()
        self.assertTrue(torch.isfinite(layer.lora_down.weight.grad).all())
        self.assertTrue(torch.isfinite(layer.lora_up.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
