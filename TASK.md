This file describes what this branch tries to do.

# Rotation Symmetric Compression (RSC) LUT

## Motivation

Lookup tables are large. We want to compress them.

## Observation

If you 1) rotate an image and 2) do super resolution, this is equivalent to 1) do super resolution 2) rotate it the same way. The same goes for mirroring. 

Therefore, we want to compress the LUT using this observation. Although LUT's input is not the full image, this observation should still apply.

## Method

The current interpolation logic is implemented in src/common/interpolation.py. It supports

- Simple LUT
- LUT with DFC

To implement my idea, it should support

- LUT with rotation symmetric compression (RSC)
- LUT with DFC and RSC

The main change is, when looking up LUT, we no longer use `a * L**3 + b * L**2 + c * L +d`. Instead, we

- Define a main element in every group of 2x2 inputs with rotation or mirror relationships. For example, we could define the main element as a<=b<=c<=d (may not be useable, check it!). 
- Just like `ref2index`, during LUT export, we generate a `rot2index` matrix. When you look up `rot2index[a,b,c,d]`, it returns `a_lookup, b_lookup, c_lookup, d_lookup` where it's the main element in that group.
- We look up `rot2index` before looking up in the real LUT.

Note that we need to generate `rot2index` and `ref2index` because we don't want to find the main element at runtime, that can be expensive.

It should be mostly the same to integrate RSC with DFC, where you only enumerate the abcd's near the diagonal, and make the `rot2index` for it to go to the main element.
