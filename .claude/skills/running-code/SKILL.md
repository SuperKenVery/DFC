---
name: running-code
description: How to run the code in this repository. You MUST read this if you want to run any python code in this repo.
---

## Prerequisites

You need to be in a `nixGLNvidia nix develop` shell. If not, you will NOT be able to solve this yourself. Tell the user to put you in such a shell.

To detect this, you can run `python -c "import pytorch; pytorch.cuda.is_available()"` to test. If this fails, immediately stop and tell the user about this problem.

## Running the code

You can run any script with python. There's direnv, so no need for `pixi run` prefix.

```
# Create an experiment
python -m run.py src.new_experiment path/to/new_exp

# Train in an experiment
CUDA_VISIBLE_DEVICES=0,1 accelerate launch run.py src.sr.1_train_model path/to/experiment
```

## Background

We're on a very old distro (ubuntu 16) and the libc is very old. Therefore, we use nix to setup a shell with newer libc. We need `nixGL` to access GPU inside the dev shell.
