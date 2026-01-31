#!/usr/bin/env python
import sys

import torch

from ..common.interpolation import InterpWithVmap, _reference_interp_single

# Simple test case
interval = 4
q = 2**interval
L = 2 ** (8 - interval) + 1
out_c = 2
upscale = 1

weight = torch.randn(L**4, out_c, upscale, upscale)

# Test a specific input
a, b, c, d = 100.0, 100.0, 100.0, 100.0  # All equal mod q

print(f"Input: a={a}, b={b}, c={c}, d={d}")
print(f"mod q: a%{q}={a % q}, b%{q}={b % q}, c%{q}={c % q}, d%{q}={d % q}")

ref_result = _reference_interp_single(weight, a, b, c, d, interval, out_c, upscale)
print(f"\nReference result shape: {ref_result.shape}")
print(f"Reference result: {ref_result}")

img_a = torch.tensor([[[[a]]]])
img_b = torch.tensor([[[[b]]]])
img_c = torch.tensor([[[[c]]]])
img_d = torch.tensor([[[[d]]]])

vmap_result = InterpWithVmap(
    weight, upscale, img_a, img_b, img_c, img_d, interval, out_c, dfc=None
)
# vmap_result has shape (B, C*out_c, ch*upscale, cw*upscale) = (1, 2, 1, 1)
# Reshape to (C, out_c, ch*upscale, cw*upscale) then select first channel
vmap_result = vmap_result[0].reshape(out_c, 1, 1)  # Shape: (2, 1, 1)

print(f"\nVmap result shape: {vmap_result.shape}")
print(f"Vmap result: {vmap_result}")

error = torch.abs(ref_result - vmap_result).max().item()
print(f"\nMax error: {error}")
print(f"Detailed diff:\n{torch.abs(ref_result - vmap_result)}")
