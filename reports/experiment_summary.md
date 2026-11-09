# HaLoRA Simulated-Noise Experiment: Corrected First-Round Results

## 1. Scope

This run is a controlled **ViT-B/16 mechanism reproduction** of HaLoRA-style
noise-aware LoRA training. It is not an exact reproduction of the paper's language-model
experiments.

The four conditions differ only in two switches:

| Condition | Training noise | Structural regularizer |
| --- | ---: | ---: |
| Vanilla LoRA | No | No |
| Noise-aware LoRA | Yes | No |
| Regularized LoRA | No | Yes |
| HaLoRA | Yes | Yes |

Shared settings include rank/alpha 4/4, 60 LoRA modules, 72 noisy frozen encoder
matrices, Gaussian noise with `sigma=0.01`, noise replacement every 400 optimizer
steps, and five fixed noisy test realizations (seeds 1-5). The regularizer uses an
unsquared Frobenius sum with `mu=0.1` only at the synchronized alignment steps 0,
400, 800, and 1200.

## 2. Final test accuracy

All values are percentages. `Noisy` is the mean over five fixed noise realizations;
the value after `+/-` is the population standard deviation. `Delta` is
`Noisy - Clean`, so a negative value means accuracy dropped under noise.

| Method | Task | Clean | Noisy | Delta |
| --- | --- | ---: | ---: | ---: |
| Vanilla LoRA | DTD | 69.15 | 69.15 +/- 0.24 | 0.00 |
| Vanilla LoRA | EuroSAT | 95.44 | 95.33 +/- 0.13 | -0.11 |
| Vanilla LoRA | dSprites Orientation | 51.69 | 49.80 +/- 0.17 | -1.89 |
| Noise-aware LoRA | DTD | 68.35 | 68.60 +/- 0.33 | +0.24 |
| Noise-aware LoRA | EuroSAT | 95.39 | 95.26 +/- 0.13 | -0.13 |
| Noise-aware LoRA | dSprites Orientation | 52.50 | 50.96 +/- 0.35 | -1.54 |
| Regularized LoRA | DTD | 69.36 | 69.17 +/- 0.34 | -0.19 |
| Regularized LoRA | EuroSAT | 94.96 | 94.74 +/- 0.17 | -0.23 |
| Regularized LoRA | dSprites Orientation | 51.82 | 50.03 +/- 0.29 | -1.79 |
| HaLoRA | DTD | 69.52 | 69.52 +/- 0.18 | 0.00 |
| HaLoRA | EuroSAT | 95.02 | 94.71 +/- 0.09 | -0.30 |
| HaLoRA | dSprites Orientation | 52.06 | 50.77 +/- 0.08 | -1.29 |

## 3. What the first round establishes

1. **The implementation path works.** All four conditions completed 100 epochs and
   1,300 optimizer steps. Noise-enabled runs used four noise maps during training
   (the initial map and replacements at steps 400, 800, and 1,200). Regularized runs
   applied the regularizer exactly four times at those same alignment steps. Every
   result contains one clean test and five
   finite noisy tests.

2. **At `sigma=0.01`, DTD and EuroSAT are only weakly noise-sensitive.** Their clean
   and noisy accuracies are almost identical. This makes them poor discriminators of
   robustness at the current noise strength.

3. **dSprites is the most noise-sensitive task.** Vanilla LoRA loses 1.89 percentage
   points under noise. Noise-aware LoRA reduces the loss to 1.54 points and improves
   noisy accuracy by 1.16 points over Vanilla LoRA. This is the clearest positive
   signal for noise-aware training in the current experiment.

4. **Periodic regularization does not collapse the LoRA branch.** Regularized LoRA is
   close to Vanilla LoRA on all three tasks: clean accuracy changes by +0.21 points on
   DTD, -0.48 on EuroSAT, and +0.13 on dSprites. The earlier severe degradation was
   entirely caused by applying the full regularizer at all 1,300 steps instead of at
   four alignment steps.

5. **HaLoRA provides its clearest benefit on DTD and dSprites.** Relative to Vanilla
   LoRA, HaLoRA improves noisy accuracy by 0.37 points on DTD and 0.96 points on
   dSprites. On dSprites, the clean-to-noisy drop decreases from 1.89 to 1.29 points,
   and the across-noise standard deviation decreases from 0.17 to 0.08 points.

6. **The benefit is not universal at this noise level.** On EuroSAT, HaLoRA is 0.62
   points below Vanilla LoRA in noisy accuracy, although its across-noise standard
   deviation is smaller (0.09 versus 0.13). At `sigma=0.01`, the accuracy differences
   are small enough that stronger-noise experiments are needed before making a broad
   robustness claim.

## 4. Observed training behavior

- Noise replacement is non-accumulating and leaves the stored clean backbone weights
  unchanged.
- Loss changes at steps 400, 800, and 1,200 confirm that new noise maps actually enter
  training. The effect is small on DTD/EuroSAT and visibly larger on dSprites.
- The structural regularizer is nonzero only at steps 0, 400, 800, and 1,200. Between
  those points, optimization uses only the noisy or clean task-loss gradient.
- The step-1,200 refresh occurs after cosine decay has made the learning rate very low.
  dSprites shows a clear loss increase at that point, but has little optimization
  capacity left to adapt to the new map.
- Best-checkpoint selection prevents the late dSprites noise-map change from replacing
  an earlier, better validation checkpoint.

## 5. Recommended next experiment

Keep these corrected 12 runs as the fixed first-round baseline. The next experiment
should expand the noise-strength sweep while preserving the corrected 400-step
alignment schedule:

```text
sigma in {0.005, 0.01, 0.02}
```

Use dSprites Orientation as the primary screening task because it is both sensitive
to the noise and shows the clearest robustness differences. After the sigma sweep,
test the interaction between the 400-step refresh schedule and cosine learning-rate
decay. A separate control can compare the sparse `mu=0.1` pulse with a continuous
regularizer whose approximate average weight is `0.1 / 400 = 2.5e-4`.

## 6. Verification record

- Python bytecode compilation: passed.
- Unit tests: 8/8 passed.
- Formal result files: 12/12 present and valid.
- LoRA/noise module counts: 60/72 in every run.
- Corrected regularized runs: `interval=400`, four regularizer applications each.
- Invalid `interval=1` runs preserved under
  `archive/invalid_regularizer_interval_1/` for audit only.
- No NaN or infinite summary metrics.
- Original `D:\baselines`: clean Git worktree at
  `eb66bddce6c08b9cd6a766a7fab9ae36df2708d9`, 32,898 files,
  2,126,542,662 bytes.
