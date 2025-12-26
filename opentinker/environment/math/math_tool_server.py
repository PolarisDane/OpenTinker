#!/usr/bin/env python3
"""Code Interpreter Math Environment Server.

This script starts a game server for math problem solving with code interpreter.
It also manages a sandbox server for Python code execution.

The server handles:
- /reset: Initialize a new math problem session
- /step: Process LLM response, extract and execute code, return results

Usage:
    # Start with auto-managed sandbox (recommended):
    python code_interpreter_math_server.py --port 8088

    # Start with external sandbox:
    python code_interpreter_math_server.py --port 8088 --sandbox-url http://localhost:8000/run_code

    # For multi-worker mode (production):
    # First start sandbox separately, then:
    uvicorn opentinker.environment.math.code_interpreter_math_server:app \\
        --host 0.0.0.0 --port 8088 --workers 4
"""

import argparse
import atexit
import time

from opentinker.environment.base_game_server import run_game_server, create_game_app
from opentinker.environment.math.math_tool_game import CodeInterpreterMathGame


# Global sandbox reference for cleanup
_sandbox_actor = None
_sandbox_url = None


def start_sandbox_background() -> str:
    """Start sandbox server in background and return URL.

    Returns:
        URL of the sandbox server
    """
    global _sandbox_actor, _sandbox_url

    import ray
    from opentinker.server.sandbox import Sandbox

    # Initialize Ray if needed
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Create and start sandbox
    _sandbox_actor = Sandbox.remote()
    ray.get(_sandbox_actor.start_server.remote())

    # Wait for server to be ready
    time.sleep(0.5)

    # Get address
    address = ray.get(_sandbox_actor.get_server_address.remote())
    _sandbox_url = f"http://{address}/run_code"

    print(f"✓ Sandbox server started at {_sandbox_url}")
    return _sandbox_url


def cleanup_sandbox():
    """Clean up sandbox server on exit."""
    global _sandbox_actor
    if _sandbox_actor is not None:
        try:
            import ray

            ray.kill(_sandbox_actor)
            print("✓ Sandbox server stopped")
        except:
            pass


# Register cleanup
atexit.register(cleanup_sandbox)


def create_app_with_sandbox(sandbox_url: str):
    """Create FastAPI app with sandbox URL configured.

    Args:
        sandbox_url: URL of the sandbox server

    Returns:
        FastAPI app
    """

    # Create game class factory with sandbox_url
    def game_factory(**kwargs):
        return CodeInterpreterMathGame(sandbox_url=sandbox_url, **kwargs)

    return create_game_app(game_class=game_factory)


def main():
    parser = argparse.ArgumentParser(description="Code Interpreter Math Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8088, help="Server port")
    parser.add_argument(
        "--sandbox-url",
        type=str,
        default=None,
        help="External sandbox URL. If not provided, starts internal sandbox.",
    )
    parser.add_argument(
        "--max-turns", type=int, default=10, help="Maximum turns per problem"
    )
    parser.add_argument(
        "--timeout", type=int, default=30, help="Sandbox execution timeout in seconds"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Code Interpreter Math Game Server")
    print("=" * 60)

    # Determine sandbox URL
    if args.sandbox_url:
        sandbox_url = args.sandbox_url
        print(f"Using external sandbox at: {sandbox_url}")
    else:
        print("Starting internal sandbox server...")
        sandbox_url = start_sandbox_background()

    print("\nServer configuration:")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Sandbox URL: {sandbox_url}")
    print(f"  Max turns: {args.max_turns}")
    print(f"  Timeout: {args.timeout}s")
    print("=" * 60 + "\n")

    # Run game server with configured sandbox
    run_game_server(
        game_class=CodeInterpreterMathGame,
        host=args.host,
        port=args.port,
        sandbox_url=sandbox_url,
        max_turns=args.max_turns,
        timeout=args.timeout,
    )


# For uvicorn multi-worker mode, create app with default sandbox URL
# Usage: Set SANDBOX_URL env var before running uvicorn
import os

_default_sandbox_url = os.environ.get("SANDBOX_URL", "http://localhost:8000/run_code")
app = create_game_app(
    game_class=CodeInterpreterMathGame,
    sandbox_url=_default_sandbox_url,
)


if __name__ == "__main__":
    main()
