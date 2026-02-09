---
name: http-pdb
description: How to use the HTTP-based debugger for agent-friendly debugging
---

## Why HTTP PDB?

This codebase uses multiprocessing, so standard pdb (which requires stdin) won't work. The `http_pdb` module provides an HTTP-based debugger that accepts commands via HTTP POST requests, making it ideal for agents and automated tools.

## Starting a Debug Session

In Python code:

```python
import http_pdb
http_pdb.set_trace()  # Starts HTTP server on port 8765 (auto-increments if busy)
```

When a breakpoint is hit, you'll see:

```
[http_pdb] Listening on port 8765
[http_pdb] Send commands:  curl --noproxy '*' localhost:8765 --data 'p variable'
[http_pdb] Exit debugger:  curl --noproxy '*' localhost:8765/exit
```

## Sending Commands

Use curl to send pdb commands (note: `--noproxy '*'` is required to bypass any proxy):

```bash
# Send a command - just POST the raw command text to /
curl --noproxy '*' localhost:8765 --data "n"
curl --noproxy '*' localhost:8765 --data "p some_var"
curl --noproxy '*' localhost:8765 --data "l"

# Common pdb commands:
# n - next line
# s - step into
# c - continue
# p <expr> - print expression
# l - list source
# w - where (stack trace)
# h - help
```

## Exiting the Debugger

To continue execution and exit the debugger:

```bash
curl --noproxy '*' localhost:8765/exit
```

## Multiprocessing Support

When used in multiprocessing contexts, each process that hits a breakpoint will automatically find a free port (starting from 8765). Check the output to see which port each debugger is listening on.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | Send pdb command (raw text body) |
| `/exit` | GET/DELETE | Exit debugger and continue execution |
