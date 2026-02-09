#!/usr/bin/env python3
"""
Example usage of http_pdb.

Run from repo root:
    python http_pdb/example.py

Then in another terminal:
    curl --noproxy '*' localhost:8765 --data "p x"
    curl --noproxy '*' localhost:8765 --data "p y"
    curl --noproxy '*' localhost:8765 --data "n"
    curl --noproxy '*' localhost:8765/exit
"""

if __name__ == "__main__":
    import sys
    import os

    # Add parent dir to path so http_pdb can be imported
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import http_pdb

    def example_function():
        x = 10
        y = 20
        z = x + y

        # Start HTTP debugger here
        http_pdb.set_trace()

        # Code continues after debugger exits
        result = z * 2
        print(f"Result: {result}")
        return result

    print("Starting example...")
    example_function()
    print("Done!")
