"""
Configuration system using TOML files with dataclasses.

This module provides a configuration system that:
1. Defines all options as dataclasses with documentation
2. Generates TOML files with comments explaining each option
3. Loads config from TOML files
"""

import argparse
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import tomllib
from fancy_dataclass import TOMLDataclass

CONFIG_FILENAME = "config.toml"


@dataclass
class ModelConfig(TOMLDataclass):
    """Model architecture configuration."""

    model: str = field(
        default="SPF_LUT_net",
        metadata={"doc": "Model architecture to use"},
    )
    nf: int = field(
        default=64,
        metadata={"doc": "Number of filters in convolutional layers"},
    )
    scale: int = field(
        default=4,
        metadata={"doc": "Upscaling factor for super-resolution"},
    )
    stages: int = field(
        default=2,
        metadata={"doc": "Number of stages in MuLUT"},
    )
    branches: int = field(
        default=3,
        metadata={"doc": "Number of parallel SR-LUT branches per stage"},
    )
    sample_size: int = field(
        default=3,
        metadata={"doc": "AutoSample layer input size (typically 3 for 3x3 input)"},
    )
    interval: int = field(
        default=4,
        metadata={"doc": "N-bit uniform sampling interval for LUT quantization"},
    )


@dataclass
class DataConfig(TOMLDataclass):
    """Data loading configuration."""

    train_dir: str = field(
        default="./data/DIV2K/",
        metadata={"doc": "Training dataset directory"},
    )
    val_dir: str = field(
        default="./data/SRBenchmark/",
        metadata={"doc": "Validation dataset directory"},
    )
    batch_size: int = field(
        default=32,
        metadata={"doc": "Training batch size"},
    )
    crop_size: int = field(
        default=48,
        metadata={"doc": "Input LR training patch size"},
    )
    worker_num: int = field(
        default=8,
        metadata={"doc": "Number of data loading workers"},
    )


@dataclass
class TrainConfig(TOMLDataclass):
    """Training configuration."""

    start_iter: int = field(
        default=0,
        metadata={"doc": "Starting iteration (0 = from scratch, >0 = resume)"},
    )
    total_iter: int = field(
        default=200000,
        metadata={"doc": "Total number of training iterations"},
    )
    lr0: float = field(
        default=1e-3,
        metadata={"doc": "Initial learning rate"},
    )
    lr1: float = field(
        default=1e-4,
        metadata={"doc": "Final learning rate (use negative for cosine-only schedule)"},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"doc": "L2 weight decay"},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"doc": "Gradient accumulation steps for effective larger batch"},
    )
    display_step: int = field(
        default=100,
        metadata={"doc": "Display training info every N iterations"},
    )
    val_step: int = field(
        default=2000,
        metadata={"doc": "Run validation every N iterations"},
    )
    save_step: int = field(
        default=2000,
        metadata={"doc": "Save checkpoint every N iterations"},
    )


@dataclass
class DFCExportConfig(TOMLDataclass):
    """DFC (Diagonal-First Compression) configuration for LUT export."""

    enabled: bool = field(
        default=False,
        metadata={"doc": "Whether to use DFC when exporting LUT"},
    )
    diagonal_width: int = field(
        default=2,
        metadata={"doc": "Diagonal width (high precision region radius)"},
    )
    sampling_interval: int = field(
        default=5,
        metadata={"doc": "Sampling interval for non-diagonal subsampling"},
    )


@dataclass
class ExportLUTConfig(TOMLDataclass):
    """LUT export configuration."""

    checkpoint_iter: int = field(
        default=200000,
        metadata={"doc": "Checkpoint iteration to export LUT from"},
    )
    dfc: DFCExportConfig = field(
        default_factory=DFCExportConfig,
        metadata={"doc": "DFC compression settings"},
    )


@dataclass
class FinetuneLUTConfig(TOMLDataclass):
    """LUT finetuning configuration."""

    export_lut_iter: int = field(
        default=200000,
        metadata={"doc": "The checkpoint iteration where LUT was exported"},
    )
    start_iter: int = field(
        default=0,
        metadata={"doc": "Starting iteration for finetuning (0 = from exported LUT)"},
    )
    total_iter: int = field(
        default=200000,
        metadata={"doc": "Total number of finetuning iterations"},
    )
    lr0: float = field(
        default=1e-4,
        metadata={"doc": "Initial learning rate for finetuning"},
    )
    lr1: float = field(
        default=1e-5,
        metadata={"doc": "Final learning rate for finetuning"},
    )
    batch_size: int = field(
        default=32,
        metadata={"doc": "Training batch size for finetuning"},
    )
    display_step: int = field(
        default=100,
        metadata={"doc": "Display training info every N iterations"},
    )
    val_step: int = field(
        default=2000,
        metadata={"doc": "Run validation every N iterations"},
    )
    save_step: int = field(
        default=2000,
        metadata={"doc": "Save checkpoint every N iterations"},
    )


@dataclass
class ExperimentConfig(TOMLDataclass):
    """Complete experiment configuration."""

    model: ModelConfig = field(
        default_factory=ModelConfig,
        metadata={"doc": "Model architecture settings"},
    )
    data: DataConfig = field(
        default_factory=DataConfig,
        metadata={"doc": "Data loading settings"},
    )
    train: TrainConfig = field(
        default_factory=TrainConfig,
        metadata={"doc": "Training settings"},
    )
    export_lut: ExportLUTConfig = field(
        default_factory=ExportLUTConfig,
        metadata={"doc": "LUT export settings"},
    )
    finetune_lut: FinetuneLUTConfig = field(
        default_factory=FinetuneLUTConfig,
        metadata={"doc": "LUT finetuning settings"},
    )


def load_config(config_path: Path) -> ExperimentConfig:
    """Load configuration from a TOML file."""
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    return ExperimentConfig.from_dict(data)


def save_config(config: ExperimentConfig, config_path: Path) -> None:
    """Save configuration to a TOML file with comments."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config.to_toml_string())


def get_default_config() -> ExperimentConfig:
    """Get the default configuration."""
    return ExperimentConfig()


@dataclass
class Experiment:
    """
    Represents a loaded experiment with its configuration and paths.

    This is the main interface for scripts to access experiment settings.
    """

    config: ExperimentConfig
    exp_dir: Path

    @property
    def checkpoint_dir(self) -> Path:
        return self.exp_dir / "checkpoints"

    @property
    def val_output_dir(self) -> Path:
        return self.exp_dir / "val"

    @property
    def log_dir(self) -> Path:
        # Use parent's logs directory
        return self.exp_dir / "tensorboard"

    @property
    def config_path(self) -> Path:
        return self.exp_dir / CONFIG_FILENAME

    def get_checkpoint_path(self, iteration: int) -> Path:
        return self.checkpoint_dir / f"checkpoint_{iteration}"

    def get_lut_checkpoint_path(self, iteration: int) -> Path:
        return self.get_checkpoint_path(iteration) / "lut"

    def get_lutft_checkpoint_path(self, iteration: int) -> Path:
        return self.exp_dir / "lutft_checkpoints" / f"checkpoint_{iteration}"

    def ensure_dirs(self) -> None:
        """Create necessary directories for the experiment."""
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.val_output_dir.mkdir(parents=True, exist_ok=True)

    def save_code(self) -> None:
        """Save a copy of the source code to the experiment directory."""
        trg_dir = self.exp_dir / "code"

        def ignore_func(directory, files):
            ignored = set(shutil.ignore_patterns("__pycache__")(directory, files))
            for f in files:
                path = os.path.join(directory, f)
                if (
                    os.path.exists(path)
                    and not os.path.isfile(path)
                    and not os.path.isdir(path)
                ):
                    ignored.add(f)
            return ignored

        shutil.copytree(
            "src",
            trg_dir / "src",
            ignore=ignore_func,
            dirs_exist_ok=True,
        )

    def print_config(self, logger) -> None:
        """Print configuration to logger."""
        logger.info("----------------- Config ---------------")
        logger.info(f"Experiment directory: {self.exp_dir}")
        logger.info(self.config.to_toml_string())
        logger.info("----------------- End -------------------")


def parse_exp_dir() -> Path:
    """Parse experiment directory from command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp-dir",
        "-e",
        type=str,
        required=True,
        help="Experiment directory containing config.toml",
    )
    args = parser.parse_args()
    return Path(args.exp_dir)


def load_experiment(exp_dir: Optional[Path] = None) -> Experiment:
    """
    Load an experiment from a directory.

    If exp_dir is None, parses it from command line arguments.
    The directory must contain a config.toml file.

    Args:
        exp_dir: Path to experiment directory, or None to parse from CLI

    Returns:
        Experiment object with loaded configuration

    Raises:
        FileNotFoundError: If config.toml doesn't exist in exp_dir
    """
    if exp_dir is None:
        exp_dir = parse_exp_dir()

    config_path = exp_dir / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Create one with: python -m src.new_experiment {exp_dir}"
        )

    config = load_config(config_path)
    return Experiment(config=config, exp_dir=exp_dir)
