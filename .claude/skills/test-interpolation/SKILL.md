---
name: test-interpolation
description: How to test the tetrahedral interpolation core logic
---

## Prerequisites

You need to be in a `nixGLNvidia nix develop` shell. If not, you will NOT be able to solve this yourself. Tell the user to put you in such a shell.

## Quick Test

```bash
python -m common.interpolation.py
```

## Understanding Test Results

The test suite compares `InterpWithVmap()` (vmap-based tetrahedral interpolation) against `_reference_interp_single()` (reference 24-case implementation from model.py).

Expected output:

```
Test complete. Max error: 0.000002, Mean error: 0.000001
```

**Tolerance**: Errors under `1e-5` are acceptable (floating point precision). Larger errors indicate a bug.

## Shape Handling

The vmap result has shape `(B, C*out_c, ch*upscale, cw*upscale)`. To compare with reference shape `(out_c, upscale, upscale)`:

```python
vmap_result = vmap_result[0].reshape(out_c, upscale, upscale)
```

## Debugging a Specific Case

```python
import sys
sys.path.insert(0, '..')
import torch
from common.interpolation import InterpWithVmap, _reference_interp_single

interval = 4
q = 2 ** interval
L = 2 ** (8 - interval) + 1
out_c = 2
upscale = 1

# Create test weight
weight = torch.randn(L**4, out_c, upscale, upscale)

# Test specific values
a, b, c, d = 100.0, 50.0, 75.0, 25.0

print(f"Testing a={a}, b={b}, c={c}, d={d}")
print(f"Fractional parts (mod {q}): {a%q:.2f}, {b%q:.2f}, {c%q:.2f}, {d%q:.2f}")

# Reference result
ref = _reference_interp_single(weight, a, b, c, d, interval, out_c, upscale)
print(f"Reference: {ref.flatten()}")

# Vmap result
img_a = torch.tensor([[[[a]]]])
img_b = torch.tensor([[[[b]]]])
img_c = torch.tensor([[[[c]]]])
img_d = torch.tensor([[[[d]]]])

vmap = InterpWithVmap(weight, upscale, img_a, img_b, img_c, img_d, interval, out_c, dfc=None)
vmap = vmap[0].reshape(out_c, upscale, upscale)
print(f"Vmap:      {vmap.flatten()}")

error = (ref - vmap).abs().max().item()
print(f"Error: {error}")
```

## Testing Full LUT Export Pipeline

To test interpolation with actual LUT exports:

```bash
cd sr/
CUDA_VISIBLE_DEVICES=0,1 \
    accelerate launch 2.5_debug_exported_lut.py \
    --model SPF_LUT_net \
    --scale 4 \
    --modes s \
    --expDir ../models/test-exportable-module-perf \
    --trainDir ../data/DIV2K \
    --valDir ../data/SRBenchmark \
    --sample-size 3 \
    --batchSize 16 \
    --startIter 22000
# Breaks at set_trace()
```

Then use remote pdb to inspect interpolation intermediates. See `remote-pdb` skill for connection details.
