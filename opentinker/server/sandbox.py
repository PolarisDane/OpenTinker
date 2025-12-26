# # Copyright 2024 Bytedance Ltd. and/or its affiliates
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.


import fastapi
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
import asyncio
import sys
import tempfile
import os
import socket
import json
import ray


@ray.remote(num_cpus=1)
class Sandbox:
    """Sandbox to execute python code."""

    def __init__(self):
        # Use localhost for single-node setups - more reliable than Ray node IP
        # which may not be accessible from all worker contexts (especially in Docker)
        self.address = "127.0.0.1"
        self.port = self._get_free_port()
        self.server_thread = None
        self.server_ready = False

    async def code_execution(self, request: Request):
        request_json = await request.json()
        code = request_json["code"]
        # print(f"execute code:\\n{code}")

        _, temp_file = tempfile.mkstemp(
            suffix=".py", prefix="temp_code", dir=None, text=True
        )
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            response = {
                "status": "Success" if process.returncode == 0 else "Failed",
                "run_result": {
                    "status": "Finished",
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "return_code": process.returncode,
                },
            }
            return JSONResponse(content=response)
        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def _get_free_port(self):
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def _run_server(self):
        """Run the FastAPI server in a separate thread."""

        app = fastapi.FastAPI()
        app.router.add_api_route("/run_code", self.code_execution, methods=["POST"])

        # Mark server as ready before starting
        self.server_ready = True

        # Run uvicorn server
        uvicorn.run(app, host="0.0.0.0", port=self.port, log_level="warning")

    def start_server(self):
        """Start the FastAPI server in a background thread."""
        import threading
        import time
        import logging

        logger = logging.getLogger(__name__)

        if self.server_thread is not None:
            logger.info(f"Sandbox server already running at {self.address}:{self.port}")
            return  # Server already started

        logger.info(f"Starting Sandbox server at {self.address}:{self.port}...")

        # Start server in background thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        # Wait for server to be ready
        max_wait = 10  # seconds
        start_time = time.time()
        while not self.server_ready and (time.time() - start_time) < max_wait:
            time.sleep(0.1)

        if not self.server_ready:
            raise RuntimeError("Sandbox server failed to start within timeout")

        # Additional wait to ensure server is actually listening
        time.sleep(0.5)

        # Verify server is actually reachable at the advertised address
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            result = sock.connect_ex((self.address, self.port))
            if result != 0:
                raise RuntimeError(
                    f"Sandbox server not listening on advertised address {self.address}:{self.port}. "
                    f"Connection test failed with code {result}"
                )
            logger.info(
                f"✓ Sandbox server successfully started and verified at {self.address}:{self.port}"
            )
        finally:
            sock.close()

    def get_server_address(self) -> str:
        """Get FastAPI server address."""
        return f"{self.address}:{self.port}"


def start_sandbox_server(host: str = "0.0.0.0", port: int = None) -> tuple:
    """Start a sandbox server and return the Ray actor and URL.

    This is a convenience function for creating a sandbox server for
    code execution in multi-turn math training.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to use (auto-select if None)

    Returns:
        Tuple of (sandbox_actor, sandbox_url) where:
        - sandbox_actor: Ray actor reference (keep alive to maintain server)
        - sandbox_url: URL of the sandbox server (e.g., "http://localhost:8000/run_code")

    Example:
        sandbox, url = start_sandbox_server()
        print(f"Sandbox running at {url}")
        # Keep sandbox alive for the duration of training
    """
    import time

    # Create sandbox actor
    sandbox = Sandbox.remote()

    # Start the server
    ray.get(sandbox.start_server.remote())

    # Wait a bit for server to be fully ready
    time.sleep(0.5)

    # Get the address
    address = ray.get(sandbox.get_server_address.remote())
    sandbox_url = f"http://{address}/run_code"

    return sandbox, sandbox_url


def create_tool_config(sandbox_url: str, output_path: str = None) -> str:
    """Create a tool config JSON file for the sandbox.

    Args:
        sandbox_url: URL of the sandbox server
        output_path: Path to save the config (auto-generated if None)

    Returns:
        Path to the created config file
    """
    import tempfile

    config = {
        "tools": [
            {
                "class_name": "opentinker.server.sandbox_tool.SandboxTool",
                "config": {"type": "native", "sandbox_fusion_url": sandbox_url},
            }
        ]
    }

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".json", prefix="tool_config_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
    else:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

    return output_path
