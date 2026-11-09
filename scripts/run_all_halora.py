"""Run all four HaLoRA conditions over the three Mini-VTAB tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = (
    "vanilla_lora",
    "noise_aware_lora",
    "regularized_lora",
    "halora",
)
TASKS = ("dtd", "eurosat", "dsprites_orientation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
        help="Run only the selected conditions (default: all four).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for condition in args.conditions:
        config_name = f"{condition}.yaml"
        for task in TASKS:
            command = [
                sys.executable,
                "-m",
                "experiments.halora.run_experiment",
                "--config",
                str(PROJECT_ROOT / "configs" / config_name),
                "--task",
                task,
                "--num-workers",
                str(args.num_workers),
            ]
            if args.smoke:
                command.append("--smoke")
            if args.overwrite:
                command.append("--overwrite")
            print(f"Running: {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
