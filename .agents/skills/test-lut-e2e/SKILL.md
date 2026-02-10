---
name: test-lut-e2e
description: Test the whole DNN network and its LUT counterpart end to end
---

# Why

Even though interpolation implementation is correct, there could still be mismatches between DNN and LUT output values.

# How to debug

First, read the `running-code` skill. This is MANDATORY.

Then, you can run `CUDA_VISIBLE_DEVICES=0,1 accelerate launch run.py src.sr.2a_debug_exported_lut -e experiment_dir` to launch a debug process.

Then, you read its stdout to know the http_pdb port. Then you use `curl` commands to debug, e.g.
```bash
curl --noproxy '*' localhost:8765 --data "cmp('.convblock1.in_c0.b0.rot0.sr')"
```

# How it works

You need to pay attention to these part of that script:

1. Where we set breakpoint
```python
dnn_output, lut_output, model_dbg, lut_dbg, cmp = run_model('butterfly', model_G, lut_model)

http_pdb.set_trace()
```

2. Where we run the DNN model, LUT model and compare output
```python
with torch.no_grad():
        dnn_output = dnn_model(test_tensor, debug_info=("", model_dbg))

    with torch.no_grad():
        lut_output = lut_model(test_tensor, debug_info=("", lut_dbg))

    def cmp(key: str, size: int = 5):
        model = model_dbg[key] * 255
        lut = lut_dbg[key] * 255
        diff = torch.abs(lut - model)
        print(f"Max diff: {diff.max()}, Mean diff: {diff.mean()}")
        return (lut - model)[0, 0, :size, :size]

    return dnn_output, lut_output, model_dbg, lut_dbg, cmp
```

3. Where we record debug outputs
   This is in `src/sr/model.py`, inside ConvBlcok
```python
for c in range(self.in_c):
  x_c = x[:, c : c + 1, :, :]
  prevx_c = prev_x[:, c : c + 1, :, :] if prev_x != None else None
  debug_dict[f"{prefix}.in_c{c}.input.x"] = x_c # <== Here!
  debug_dict[f"{prefix}.in_c{c}.input.prev_x"] = prevx_c  # <== Here!
  pred = 0
  for b in range(self.branches):
      pad = self.sample_size - 1
      key = "DepthwiseBlock{}_{}".format(c, b)
      sub_module = getattr(self, key)
      for r in [0, 1, 2, 3]:
          y = ...

          assert (
              isinstance(pred, torch.Tensor) == False or pred.shape == y.shape
          ), f"Unexpected shape: pred={pred.shape}, y={y.shape}"
          pred += y
          debug_dict[f"{prefix}.in_c{c}.b{b}.rot{r}"] = y / 127 # <== Here!

  x_out += pred
  debug_dict[f"{prefix}.in_c{c}"] = pred / (self.branches * 4 * 127) # <== Here!
```
