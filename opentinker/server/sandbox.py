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
import re
import aiohttp
from verl.tools.base_tool import BaseTool, OpenAIFunctionToolSchema, ToolResponse
from transformers.utils import get_json_schema

import requests

import fastapi
import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse
from pprint import pprint
import asyncio
import sys
import tempfile
import os
import socket
import json
import argparse
import ray


# class Sandbox:
#     """Sandbox to execute python code."""

#     def __init__(self, host: str = "0.0.0.0", port: int = None):
#         self.host = host
#         self.port = port if port is not None else self._get_free_port()
#         self.app = self._create_app()

#     def _create_app(self) -> fastapi.FastAPI:
#         """Create and configure FastAPI application."""
#         app = fastapi.FastAPI()
#         app.router.add_api_route("/run_code", self.code_execution, methods=["POST"])
#         return app

#     async def code_execution(self, request: Request):
#         request_json = await request.json()
#         code = request_json["code"]
#         # print(f"execute code:\n{code}")

#         _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
#         with open(temp_file, "w") as f:
#             f.write(code)

#         try:
#             process = await asyncio.create_subprocess_exec(
#                 sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
#             )

#             stdout, stderr = await process.communicate()

#             response = {
#                 "status": "Success" if process.returncode == 0 else "Failed",
#                 "run_result": {
#                     "status": "Finished",
#                     "stdout": stdout.decode(),
#                     "stderr": stderr.decode(),
#                     "return_code": process.returncode,
#                 },
#             }
#             return JSONResponse(content=response)
#         finally:
#             try:
#                 os.unlink(temp_file)
#             except Exception:
#                 pass

#     def _get_free_port(self):
#         with socket.socket() as sock:
#             sock.bind(("", 0))
#             return sock.getsockname()[1]

#     def run(self):
#         """Start the FastAPI server."""
#         print(f"Starting Sandbox server at {self.host}:{self.port}")
#         uvicorn.run(self.app, host=self.host, port=self.port, log_level="warning")

#     def get_server_address(self) -> str:
#         """Get FastAPI server address."""
#         return f"{self.host}:{self.port}"


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run Sandbox code execution server")
#     parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server to")
#     parser.add_argument("--port", type=int, default=None, help="Port to bind the server to (default: auto-assign)")
#     args = parser.parse_args()
    
#     sandbox = Sandbox(host=args.host, port=args.port)
#     print(f"Sandbox server address: {sandbox.get_server_address()}")
#     sandbox.run()

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

        _, temp_file = tempfile.mkstemp(suffix=".py", prefix="temp_code", dir=None, text=True)
        with open(temp_file, "w") as f:
            f.write(code)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_file, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
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
        import threading
        
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
            logger.info(f"✓ Sandbox server successfully started and verified at {self.address}:{self.port}")
        finally:
            sock.close()

    def get_server_address(self) -> str:
        """Get FastAPI server address."""
        return f"{self.address}:{self.port}"