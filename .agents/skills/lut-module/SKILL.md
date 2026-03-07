---
name: lut-module
description: How the ExportableLUTModule system works - exporting DNNs to LUTs, loading, saving, forward dispatch, and finetuning.
---

# Overview

`src/common/lut_module.py` defines `ExportableLUTModule`, a base class that lets a single model definition serve as both a DNN (for training) and a LUT (for inference/finetuning). The key idea: instead of maintaining two separate model classes, you inherit from `ExportableLUTModule` and override a few methods.

# Class Hierarchy

Every module from the root model down to the LUT-exportable leaf must inherit `ExportableLUTModule`. Non-exportable siblings can be plain `nn.Module`.

```
SPF_LUT_net (ExportableLUTModule)  -- root model
  |- ConvBlock (ExportableLUTModule)  -- contains exportable children
  |    |- DepthwiseBlock (ExportableLUTModule)  -- contains exportable children
  |         |- MuLUTConv (ExportableLUTModule)  -- leaf, actually exported to LUT
  |              |- MuLUTUnit (ExportableLUTModule)  -- the actual conv layers
  |- MuLUTcUnit (ExportableLUTModule)
  ...
```

# Forward Dispatch

`ExportableLUTModule.__call__` (`lut_module.py:124`) checks:

```python
if self.lut_weight is not None and self.lut_config is not None:
    return self.lut_forward(...)   # LUT interpolation
else:
    return self.forward(...)       # normal DNN forward
```

This is **not** controlled by a context manager. Once `lut_weight` and `lut_config` are set (by `load_from_lut`), they persist and all future calls use `lut_forward`. This means you must always use `submodule(x)` rather than `submodule.forward(x)`, otherwise the dispatch is bypassed.

# The Three Operations

## 1. Export DNN to LUT (enumerate inputs, store outputs)

Used in `2_compress_lut_from_net.py`. Context manager redirects `state_dict()` to call `export_to_lut()` on each module instead of `_save_to_state_dict()`.

```python
with model.save_as_lut(lut_cfg):
    state_dict = model.state_dict()     # or accelerator.save_model(...)
```

Under the hood: sets class variable `ExportableLUTModule.redirect_state_dict = cfg`. Each module's `_save_to_state_dict` checks this and calls `export_to_lut` instead. The leaf module (`MuLUTUnit` in `network.py`) enumerates all quantized inputs, runs DNN forward, and stores the output as `lut_weight` (int8).

## 2. Load LUT state into model

Used in both `2_compress_lut_from_net.py` (for validation) and `3_finetune_compress_lut.py` (for finetuning). Context manager redirects `load_state_dict()` to call `load_from_lut()` on each module instead of `_load_from_state_dict()`.

```python
with model.load_state_from_lut(lut_cfg, accelerator):
    model.load_state_dict(state_dict)
```

Under the hood: sets class variable `ExportableLUTModule.redirect_load_state_dict = (cfg, accelerator)`. The leaf module's `load_from_lut` (`network.py:283`) sets `self.lut_weight = nn.Parameter(...)` and `self.lut_config = cfg`, which permanently switches forward dispatch to `lut_forward`.

## 3. Save LUT state (for checkpointing during finetuning)

Same as export but from an already-LUT model. Uses the same context manager:

```python
with model.save_as_lut(lut_cfg):
    accelerator.save_model(model, save_path)
```

# Finetuning LUT (critical ordering)

When finetuning LUT values as trainable parameters, the order of operations matters:

```python
# 1. Create the model
model_G = SPF_LUT_net(...)

# 2. Load LUT state FIRST -- this creates lut_weight as nn.Parameter
with model_G.load_state_from_lut(lut_cfg, accelerator):
    model_G.load_state_dict(state_dict)

# 3. THEN create optimizer -- so it picks up lut_weight, not DNN weights
params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
opt_G = optim.Adam(params_G, ...)

# 4. Then accelerator.prepare
model_G, opt_G, ... = accelerator.prepare(model_G, opt_G, ...)
```

If you create the optimizer before loading LUT state, the optimizer references the old DNN parameters (conv weights). The forward uses `lut_forward` correctly, but gradients update the wrong (unused) parameters.

# block_submodule_state_load_save

If a module overrides `export_to_lut` or `load_from_lut` to handle its own serialization (replacing its children's weights with a single LUT), it must:

1. Call `self.block_submodule_state_load_save()` in `__init__`
2. Call `self.export_to_lut_post_hook()` at the end of `export_to_lut`
3. Call `self.load_from_lut_post_hook()` at the end of `load_from_lut`

This temporarily hides `self._modules` so `state_dict`/`load_state_dict` doesn't recurse into children (whose weights are now replaced by the LUT).

# Key Files

- `src/common/lut_module.py` -- `ExportableLUTModule` base class, input enumeration utilities
- `src/common/network.py` -- Concrete `export_to_lut`, `load_from_lut`, `lut_forward` implementations (`MuLUTUnit`, `MuLUTConv`)
- `src/common/interpolation.py` -- Tetrahedral interpolation used in `lut_forward`
- `src/sr/2_compress_lut_from_net.py` -- Export script (DNN -> LUT)
- `src/sr/3_finetune_compress_lut.py` -- Finetune script (optimize LUT values)
