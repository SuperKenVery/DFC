---
name: manage-experiments
description: How to create an experiment, and train, export to LUT, finetune LUT in it.
---

## Prerequisites

See the `running-code` skill.

## Configuration System

The project uses TOML configuration files (`config.toml`) to manage experiment settings. All settings are defined in typed dataclasses in `src/common/config.py`, providing full type checking and IDE support.

### Creating a New Experiment

Use the experiment wizard to create a new experiment with a config file:

```bash
python -m src.new_experiment models/my_experiment
```

This creates `models/my_experiment/config.toml` with all available options and documentation comments.

### Config File Structure

The `config.toml` file has the following sections:

- `[model]` - Model architecture settings (model type, scale, modes, stages, etc.)
- `[data]` - Data loading settings (directories, batch size, crop size, workers)
- `[train]` - Training settings (iterations, learning rate, checkpointing)
- `[export_lut]` - LUT export settings (checkpoint iteration, DFC compression)
- `[finetune_lut]` - LUT finetuning settings (separate lr, iterations for finetuning)

### Using Config Files

Once you have a config.toml, just specify the experiment directory with `--exp-dir` or `-e`:

```bash
# Training
accelerate launch run.py src.sr.1_train_model -e models/my_experiment

# Export LUT (uses export_lut.checkpoint_iter from config)
accelerate launch run.py src.sr.2_compress_lut_from_net -e models/my_experiment

# Finetune LUT (uses finetune_lut section from config)
accelerate launch run.py src.sr.3_finetune_compress_lut -e models/my_experiment
```
