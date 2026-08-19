import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kl_probe.engine import wait_healthy


class _Handler(BaseHTTPRequestHandler):
    health_status = 200

    def do_GET(self):
        status = self.health_status if self.path == "/health" else 404
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_wait_healthy_returns_on_200(http_server):
    _Handler.health_status = 200
    asyncio.run(wait_healthy(http_server, timeout_s=5.0, interval_s=0.1))


def test_wait_healthy_fails_fast_when_port_is_squatted(http_server):
    _Handler.health_status = 404
    with pytest.raises(RuntimeError, match="another service"):
        asyncio.run(wait_healthy(http_server, timeout_s=30.0, interval_s=0.1))


def test_wait_healthy_times_out_on_non_200(http_server):
    _Handler.health_status = 503  # still loading — retryable, not fatal
    with pytest.raises(TimeoutError):
        asyncio.run(wait_healthy(http_server, timeout_s=0.3, interval_s=0.1))
