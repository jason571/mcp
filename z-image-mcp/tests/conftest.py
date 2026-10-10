import asyncio
import hashlib
import socket
import time

import pytest
import uvicorn
from PIL import Image

from z_image_mcp.backend import check_canceled
from z_image_mcp.config import Settings
from z_image_mcp.server import create_app


class FakeBackend:
    """Deterministic CPU double, never used in the deployed service."""

    loaded = False

    def __init__(self, delay=0.03):
        self.delay = delay
        self.requests = []
        self.active = 0
        self.max_active = 0

    def unload(self):
        self.loaded = False

    def generate(self, request, output, progress, cancel):
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        self.requests.append(request)
        self.loaded = True
        try:
            if request.prompt == "fail":
                raise RuntimeError("simulated CUDA failure")
            for step in range(1, 4):
                check_canceled(cancel)
                progress(step * 25, "denoising", f"step {step}/3")
                time.sleep(self.delay)
            check_canceled(cancel)
            Image.new("RGB", (request.width, request.height), (128, 80, 190)).save(output)
            return {
                "width": request.width,
                "height": request.height,
                "seed": request.seed,
                "mime_type": "image/png",
                "bytes": output.stat().st_size,
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        finally:
            self.active -= 1


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, min_free_disk_mb=0)


@pytest.fixture
async def live_server(settings):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    settings.port = port
    settings.public_url = f"http://127.0.0.1:{port}"
    backend = FakeBackend()
    app = create_app(settings, backend)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield settings.public_url, app, backend
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 15)
        sock.close()
