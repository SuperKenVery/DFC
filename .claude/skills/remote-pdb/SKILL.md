---
name: remote-pdb
description: How to connect to a remote pdb session over TCP
---

## Why Remote PDB?

This codebase uses multiprocessing, so standard pdb (which requires stdin) won't work. Instead, it uses `remote_pdb` which listens on a TCP port for commands.

## Connection Method

Use named pipes to communicate with the remote pdb session. This is safer than using nc repeatedly and handles disconnections gracefully.

### Setup (One-time)

```bash
cd /data/xyh/DFCs/speedup/sr

# Create named pipes
mkfifo /tmp/pdb_input 2>/dev/null || true
```

### Connect to Running Session

When a breakpoint is hit, the debugger outputs something like:

```
Remote debugger listening on 127.0.0.1:XXXX
```

Create a persistent connection using tail, tee, and nc:

```bash
# Terminal 1: Connect to pdb and save output to a file
tail -f /tmp/pdb_input | nc 127.0.0.1 XXXX | tee /tmp/pdb_output &

# Terminal 2: Send commands
echo "h" > /tmp/pdb_input

# Terminal 3 (or after waiting): Read the persistent output file
cat /tmp/pdb_output
```

**Why `tee`?** FIFOs don't buffer data - it's lost if you don't read it in real-time. Using `tee` writes to a regular file so you can read it later.

## Cleanup

```bash
# Kill the nc/tail processes
killall nc tail 2>/dev/null || true

# Remove pipes and output files
rm /tmp/pdb_input /tmp/pdb_output 2>/dev/null || true
```
