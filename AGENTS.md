# Image super-resolution accelerated with lookup-tables

This is the code for an image super-resolution model.

## Model

1. It was at first SR-LUT, a convolution network exported to a lookup table.

Different from a normal convolution, where you apply the first conv kernel and the image became a little bit smaller, then you apply the second conv kernel on the whole output, and it becomes smaller again. In SR-LUT, the conv kernel is 2x2. You feed 2x2 pixels to the first conv kernel, and it outputs tensor `x`, and you directly feed `x` to the second conv kernel and so on. In this way, the input of the module is limited enough that it's possible to enumerate its all possible input.

In order to enlarge input size without exploding the LUT size, image was rotated into 4 directions (0, 90, 180, 270), fed into SR-LUT, then averaged to produce the result.

To make the LUT size useable, they did 4bit quantization. So instead of enumerating from 0 to 255, they enumerate 0, 17, 34, etc. and in inference, do interpolation.

2. Then it was MuLUT, which basically just
    1. Put three SR-LUT in parallel. Let's call this a LUT Group. The LUT Group has a 3x3 input. Each SR-LUT still has a 2x2 input, the four pixels are picked from the 9 input pixels. Three SR-LUT had different picking pattern, which when overlayed together, can cover the full 3x3 input. The three picking pattern are called s,d,y.
    2. Put two LUT Group in serie. The first LUT Group didn't to SR in order to increase effective input size.

3. Then there was DFC, diagonal-first compression. The 3x3 pixels are a tiny area in the image, so their colors are probably similar. Therefore, only the diagonal of the LUT was important. It compressed the rest of LUT more aggressively, resulting in much smaller storage size.

To make use of the compressed size, they stacked 4 LUT Group together, added a channel conv (channel wise lut group), then used a upblock (4 lut group, corresponding to 4 input channel, then average output). This resulted in even better SR performance.

4. Then there was AutoLUT. Which:
    1. Instead of manually designing the pattern from 3x3, it uses an AutoSample layer to do a weighed average of the 9 pixels, for 4 times, to produce a 2x2 input for SR-LUT. This actually eliminated the need for 3 parallel SR-LUT with different sample pattern. Currently, the letters passed to `--modes` config doesn't patter, only the number of letters. So `--modes ss` is same as `--modes yy` (both are two SR-LUT with AutoSample), and `--modes s` means only one SR-LUT with AutoSample in a LUT Group.
    2. It introduced residual connection named AdaRL. If you directly feed `SR-LUT(x + prev_x)`, the range would go from 0-255 to 0-510 which explodes LUT size. AdaRL introduced learnable weights for the 4 samples pixels, and did a weighed average of x and prev_x.

## Code structure

The model was trained to do many things, like super-resolution, denoising, deblurring, deblocking etc. You can see them in `src/dblocking/`, `src/dblur/`, `src/dnoising/` and `src/sr/`.

The contents in these folders are actually highly similar. We currently only focus on `src/sr/`.

- 1_train_model.py: Train the image resolution DNN
- 2_compress_lut_from_net.py: Enumerate the input and export a DNN into LUT.
- 3_finetune_compress_lut.py: Treat Look up table values as optimizable parameters in a DNN, and use pytorch to optimize it. This is needed because we have quantization during export.
- 4_compress_lut_from_lut.py: Previously used for DFC, not needed now.
- 4_test_SPF-LUT_DFC.py: Unused.
- data.py: data loaders.
- debug.ipynb: Don't care about it.
- model.py: Defines whole models. This includes:
    - The DNN model for training (SPF_LUT_net). `ConvBlock` is the LUT Group.
    - The LUT model for finetuning (SPF_LUT_DFC)
    - However, this is obviously non-sustainable as you have to maintain two set of code. Therefore, I am taking a new approach in `lut_module.py` that allows you to share a single model definition between the DNN model and LUT model. Therefore, the LUT definitioin in model.py is not used anymore.
- train_utils.py: utilities

In `src/common/`:

- config.py: TOML-based configuration system with typed dataclasses. This is the primary way to configure experiments.
- interpolation.py: A clean implementation of interpolation using `torch.vmap`.
- lut_module.py: This is a util class that helps exporting any network to LUTs. In a LUT network, you typically have a part of it you want to export to LUT (the SR-LUT part), and some other parts stayed the same (the AutoSample and AdaRL part). When exporting to LUT, we call recursively like `state_dict`, but only export the exportable parts. When forwarding, we dispatch between DNN forward and LUT forward.
- network.py: Building blocks for SR networks. `src/sr/model.py` uses many modules here.
- option.py: DEPRECATED - Legacy CLI options, kept for backward compatibility with non-SR tasks.
- utils.py: util
- vmap_helper.py: Workaround pytorch vmap bugs
- Writer.py: tensorboard writer

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

All settings come from the config file. There is no CLI override - edit the config.toml to change settings.

## Running Scripts

All scripts are run using the `run.py` entry point with accelerate:

```bash
# Create new experiment with config
python -m src.new_experiment models/my_experiment

# Edit config.toml to customize settings, then:

# SR training
accelerate launch run.py src.sr.1_train_model -e models/my_experiment

# SR LUT export
accelerate launch run.py src.sr.2_compress_lut_from_net -e models/my_experiment

# SR LUT finetuning
accelerate launch run.py src.sr.3_finetune_compress_lut -e models/my_experiment

# Multi-GPU training
accelerate launch --multi_gpu --num_processes=4 run.py src.sr.1_train_model -e models/my_experiment
```

The project uses proper Python package imports with relative imports (e.g., `from ..common.network import *`).

## Acting guidelines

- Do NOT commit your changes. I'll manage the git history.

## Updating

If this file goes out of sync with source code, remember to update it!
