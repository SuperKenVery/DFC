#!/usr/bin/env python
"""
Generic entry point for running any module in src/ with accelerate launch.

Usage:
    accelerate launch run.py src.sr.1_train_model --your-args
    accelerate launch run.py src.dblocking.1_train_model --your-args
    accelerate launch run.py src.sr.2_compress_lut_from_net --your-args

Multi-GPU:
    accelerate launch --multi_gpu --num_processes=4 run.py src.sr.1_train_model --args
"""

import os
import runpy
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: accelerate launch run.py <module_path> [args...]")
        print("\nExamples:")
        print("  accelerate launch run.py src.sr.1_train_model --scale 4 --modes ss")
        print(
            "  accelerate launch run.py src.sr.2_compress_lut_from_net --model SPF_LUT_net"
        )
        print("  accelerate launch run.py src.dblocking.1_train_model --qf 10")
        print("\nMulti-GPU:")
        print(
            "  accelerate launch --multi_gpu --num_processes=4 run.py src.sr.1_train_model"
        )
        sys.exit(1)

    # Get the module to run
    module_name = sys.argv[1]

    # Remove the module name from sys.argv so the target script sees correct args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    # Run the module
    try:
        runpy.run_module(module_name, run_name="__main__")
    except ModuleNotFoundError as e:
        print(f"Error: Could not find module '{module_name}'")
        print(f"Details: {e}")
        print("\nMake sure the module path is correct (e.g., src.sr.1_train_model)")
        sys.exit(1)
