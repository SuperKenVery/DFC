"""
HTTP-based Python debugger.

Usage:
    import http_pdb
    http_pdb.set_trace()  # Starts HTTP server (auto-selects port if busy)

Then send commands via HTTP:
    curl --noproxy '*' localhost:8765 --data "n"
    curl --noproxy '*' localhost:8765 --data "p some_var"
    curl --noproxy '*' localhost:8765/exit  # Exit debugger and continue
"""

from .server import set_trace, HttpPdb

__all__ = ['set_trace', 'HttpPdb']
