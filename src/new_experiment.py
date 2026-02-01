#!/usr/bin/env python
"""
Experiment Wizard - Initialize a new experiment with a config.toml file.

Usage:
    python -m src.new_experiment <experiment_path>

    # Or with accelerate:
    accelerate launch run.py src.new_experiment <experiment_path>

This creates a new experiment directory with a config.toml file containing
all available options with their default values and documentation comments.
"""

import argparse
import sys
from pathlib import Path

from .common.config import ExperimentConfig, save_config


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a new experiment with a config.toml file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a new experiment
    python -m src.new_experiment models/my_experiment

    # The created config.toml can then be edited and used:
    accelerate launch run.py src.sr.1_train_model -e models/my_experiment
""",
    )
    parser.add_argument(
        "experiment_path",
        type=str,
        help="Path to the new experiment directory",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing config.toml if it exists",
    )

    args = parser.parse_args()

    exp_path = Path(args.experiment_path)
    config_path = exp_path / "config.toml"

    # Check if directory exists
    if not exp_path.exists():
        exp_path.mkdir(parents=True)
        print(f"Created experiment directory: {exp_path}")
    else:
        print(f"Using existing directory: {exp_path}")

    # Check if config already exists
    if config_path.exists() and not args.force:
        print(f"Error: {config_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    # Create default config
    config = ExperimentConfig()

    # Save config with comments
    save_config(config, config_path)

    print(f"Created config file: {config_path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {config_path} to customize your experiment")
    print(
        f"  2. Run training: accelerate launch run.py src.sr.1_train_model -e {exp_path}"
    )
    print(
        f"  3. Export LUT:   accelerate launch run.py src.sr.2_compress_lut_from_net -e {exp_path}"
    )
    print(
        f"  4. Finetune LUT: accelerate launch run.py src.sr.3_finetune_compress_lut -e {exp_path}"
    )


if __name__ == "__main__":
    main()
