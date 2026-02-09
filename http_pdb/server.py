"""
HTTP-based Python debugger server.

Simple API for agent-friendly debugging:
- POST / with raw command body -> returns output
- GET /exit or DELETE /exit -> exit debugger
"""

import pdb
import sys
import io
import socket
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional


def find_free_port(start_port: int = 8765, max_attempts: int = 100) -> int:
    """Find a free port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port in range {start_port}-{start_port + max_attempts}")


class QueuedPdb(pdb.Pdb):
    """A Pdb instance that reads commands from a queue instead of stdin."""

    def __init__(self):
        self.output_buffer = io.StringIO()
        self.command_ready = threading.Event()
        self.pdb_ready = threading.Event()
        self.should_exit = False
        self._pending_command: Optional[str] = None
        super().__init__(stdin=self, stdout=self.output_buffer)

    def readline(self) -> str:
        """Called by pdb when it needs input."""
        # Signal that pdb is ready for a command
        self.pdb_ready.set()

        # Wait for a command to be available
        self.command_ready.wait()
        self.command_ready.clear()

        if self.should_exit:
            return "c\n"  # continue command to exit

        cmd = self._pending_command or ""
        self._pending_command = None
        return cmd + "\n"

    def send_command(self, cmd: str) -> str:
        """Send a command to pdb and get the output."""
        # Wait for pdb to be ready
        if not self.pdb_ready.wait(timeout=5):
            return "Error: pdb not ready (timeout)"

        # Clear output buffer
        self.output_buffer.seek(0)
        self.output_buffer.truncate(0)

        # Send command
        self._pending_command = cmd
        self.pdb_ready.clear()
        self.command_ready.set()

        # Wait for pdb to process and be ready again
        if not self.pdb_ready.wait(timeout=30):
            return self.output_buffer.getvalue() + "\n(command timed out)"

        return self.output_buffer.getvalue()

    def exit_debugger(self) -> str:
        """Signal the debugger to exit and continue execution."""
        self.should_exit = True
        self.command_ready.set()
        return "Continuing execution...\n"


class HttpPdb:
    """HTTP server wrapper for pdb."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.requested_port = port
        self.port: int = 0  # Will be set when server starts
        self.pdb_instance: Optional[QueuedPdb] = None
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self._shutdown = False

    def _create_handler(self):
        """Create request handler with access to pdb instance."""
        pdb_ref = self

        class PdbHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging

            def do_GET(self):
                if self.path == '/exit':
                    msg = pdb_ref.pdb_instance.exit_debugger()
                    pdb_ref._shutdown = True
                    self._respond(msg)
                else:
                    self._respond("POST command to / or GET /exit\n", 404)

            def do_DELETE(self):
                if self.path == '/exit':
                    self.do_GET()
                else:
                    self._respond("Not found\n", 404)

            def do_POST(self):
                if self.path == '/' or self.path == '':
                    length = int(self.headers.get('Content-Length', 0))
                    cmd = self.rfile.read(length).decode('utf-8').strip()

                    if not cmd:
                        self._respond("No command provided\n", 400)
                        return

                    output = pdb_ref.pdb_instance.send_command(cmd)
                    self._respond(output)
                else:
                    self._respond("POST to /\n", 404)

            def _respond(self, text: str, code: int = 200):
                body = text.encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', 'text/plain')
                self.send_header('Content-Length', len(body))
                self.end_headers()
                self.wfile.write(body)

        return PdbHandler

    def _run_server(self):
        """Run the HTTP server."""
        self.port = find_free_port(self.requested_port)
        self.server = HTTPServer((self.host, self.port), self._create_handler())
        self.server.timeout = 0.5

        # Print connection info
        print(f"\n[http_pdb] Listening on port {self.port}")
        print(f"[http_pdb] Send commands:  curl --noproxy '*' localhost:{self.port} --data 'p variable'")
        print(f"[http_pdb] Exit debugger:  curl --noproxy '*' localhost:{self.port}/exit\n")
        sys.stdout.flush()

        while not self._shutdown:
            self.server.handle_request()

        self.server.server_close()

    def set_trace(self, frame=None):
        """Start debugging at the current frame."""
        if frame is None:
            frame = sys._getframe().f_back

        self.pdb_instance = QueuedPdb()

        # Start HTTP server in background
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        # Give server a moment to start
        import time
        time.sleep(0.1)

        # Start pdb - this blocks until 'c' (continue) is issued
        self.pdb_instance.set_trace(frame)


# Global instance for convenience
_http_pdb: Optional[HttpPdb] = None


def set_trace(host: str = "0.0.0.0", port: int = 8765):
    """
    Start HTTP-based debugger at the current frame.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to listen on (default: 8765, auto-increments if in use)
    """
    global _http_pdb
    _http_pdb = HttpPdb(host=host, port=port)
    _http_pdb.set_trace(sys._getframe().f_back)
