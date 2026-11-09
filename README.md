# HaLoRA Simulated Noise Experiment

This project tests the HaLoRA mechanism on a controlled ViT-B/16 transfer-learning
benchmark. It contains a standalone copy of the required model and data-pipeline
source while excluding local datasets, pretrained weights, checkpoints, and the
read-only baseline project from version control.

This is a **mechanism reproduction on ViT**, not an exact reproduction of the paper's
Qwen/LLaMA experiments.

## Four controlled conditions

| Condition | Training noise | Structural regularizer |
| --- | ---: | ---: |
| `vanilla_lora` | No | No |
| `noise_aware_lora` | Yes | No |
| `regularized_lora` | No | Yes |
| `halora` | Yes | Yes |

All other settings are identical across the four YAML files in `configs/`.

## Locked experiment definition

- backbone: `google/vit-base-patch16-224-in21k`;
- tasks: DTD, EuroSAT, and dSprites Orientation;
- LoRA targets in every encoder block: Query, Key, Value, MLP `fc1`, MLP `fc2`;
- LoRA rank/alpha: 4/4, giving a scale of 1;
- LoRA modules: 60;
- noisy frozen encoder matrices: Q, K, V, attention output, `fc1`, and `fc2`;
- noisy modules: 72;
- noise: elementwise i.i.d. Gaussian within 64x64 blocks, scaled by each block's
  maximum absolute clean weight;
- training/test sigma: 0.01;
- training noise is replaced, never accumulated, every 400 optimizer steps;
- regularizer: unsquared Frobenius/Euclidean Gram norm, summed over all LoRA modules;
- regularizer weight: 0.1, applied only at the synchronized alignment steps
  0, 400, 800, and 1200;
- test: clean once and noisy under fixed hardware maps with seeds 1-5.

For stored PyTorch LoRA weights `D` (`rank x input`) and `U` (`output x rank`),
the regularizer is computed efficiently as

```text
||D D^T||_F + ||U^T U||_F
```

which is equal to the direct large-Gram formulation but constructs only rank-by-rank
matrices.

## Verification

Install the Python dependencies and prepare the excluded assets:

```powershell
python -m pip install -r requirements.txt
python -m scripts.download_pretrained_model
python -m scripts.download_mini_vtab
python -m scripts.prepare_dsprites_orientation
```

Run the unit tests:

```powershell
E:\anaconda\python.exe -m unittest discover -s tests -v
```

Run one short smoke experiment:

```powershell
E:\anaconda\python.exe -m experiments.halora.run_experiment `
  --config configs/halora.yaml --task dtd --smoke
```

Run all four conditions over all three tasks:

```powershell
E:\anaconda\python.exe -m scripts.run_all_halora
```

Rerun only the two regularized conditions:

```powershell
E:\anaconda\python.exe -m scripts.run_all_halora `
  --conditions regularized_lora halora --overwrite
```

Checkpoints and JSON results are written under `checkpoints/<condition>/...` and
`results/<condition>/...`. The original `D:\baselines` directory is never modified.

## Completed corrected first-round results

The 12 formal runs (four conditions by three tasks) have been completed and verified.
The regularized conditions use the corrected paper schedule with four alignment
events, not per-step regularization. The superseded `interval=1` runs are retained
under `archive/invalid_regularizer_interval_1/` for audit only.
See [`reports/experiment_summary.md`](reports/experiment_summary.md) for the accuracy
table, interpretation, limitations, and recommended next experiment.
