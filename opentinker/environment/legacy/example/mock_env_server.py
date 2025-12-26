#!/usr/bin/env python3
"""Mock Environment Server for Testing GenericAgentLoop.

This script provides a simple HTTP server that simulates a Gym-like environment.
It exposes /reset and /step endpoints for GenericAgentLoop to interact with.

Usage:
    python mock_env_server.py --port 8080

The server implements a simple text-based game where the agent must find a treasure.
"""

import argparse
import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any


class MockEnvironment:
    """Simple text adventure environment for testing."""

    def __init__(self):
        self.instances: dict[str, dict[str, Any]] = {}
        self.target_word = "treasure"

    def reset(self, instance_id: str, **kwargs) -> str:
        """Reset the environment and return initial observation."""
        self.instances[instance_id] = {
            "step_count": 0,
            "found_treasure": False,
            "hints_given": 0,
        }
        return (
            "You are in a dark cave. You must find the treasure. "
            "Type 'search' to look around, 'go north/south/east/west' to move, "
            "or describe what you want to do."
        )

    def step(self, instance_id: str, action: str) -> tuple[str, float, bool, dict]:
        """Execute action and return (observation, reward, done, info)."""
        if instance_id not in self.instances:
            return "Error: Instance not found.", 0.0, True, {}

        state = self.instances[instance_id]
        state["step_count"] += 1
        action_lower = action.lower().strip()

        # Check for treasure mention
        if self.target_word in action_lower:
            state["found_treasure"] = True
            return (
                "Congratulations! You found the treasure! The cave is filled with gold and jewels.",
                10.0,  # Big reward for finding treasure
                True,  # Episode done
                {"success": True},
            )

        # Handle different actions
        if "search" in action_lower:
            state["hints_given"] += 1
            hints = [
                "You see a faint glimmer to the north.",
                "You hear water dripping nearby. The sound echoes strangely.",
                "There are ancient markings on the wall pointing east.",
                "You notice the ground is softer here, as if recently disturbed.",
            ]
            hint = hints[min(state["hints_given"] - 1, len(hints) - 1)]
            return hint, 0.1, False, {}

        elif any(d in action_lower for d in ["north", "south", "east", "west"]):
            direction = next(
                d for d in ["north", "south", "east", "west"] if d in action_lower
            )
            observations = {
                "north": "You move north. The passage becomes narrower. You see something shiny ahead.",
                "south": "You move south. The air becomes fresher. You might be near an exit.",
                "east": "You move east. You enter a chamber with strange symbols on the walls.",
                "west": "You move west. The darkness deepens. You hear distant rumbling.",
            }
            return observations[direction], 0.0, False, {}

        elif "help" in action_lower:
            return (
                "Commands: 'search' to look around, 'go north/south/east/west' to move. "
                "Your goal is to find the treasure!",
                0.0,
                False,
                {},
            )

        else:
            # Generic response for other actions
            responses = [
                "You try that, but nothing special happens.",
                "Interesting idea. The cave remains silent.",
                "You continue exploring. The shadows seem to shift.",
            ]
            return random.choice(responses), -0.1, False, {}


class MockEnvHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the mock environment."""

    env = MockEnvironment()

    def _send_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_response({"error": "Invalid JSON"}, 400)
            return

        instance_id = data.pop("instance_id", "default")  # Use pop to remove from data

        if self.path == "/reset":
            observation = self.env.reset(instance_id, **data)
            self._send_response({"observation": observation})

        elif self.path == "/step":
            action = data.get("action", "")
            observation, reward, done, info = self.env.step(instance_id, action)
            self._send_response(
                {
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )

        elif self.path == "/health":
            self._send_response({"status": "healthy"})

        else:
            self._send_response({"error": f"Unknown endpoint: {self.path}"}, 404)

    def do_GET(self):
        if self.path == "/health":
            self._send_response({"status": "healthy"})
        else:
            self._send_response({"error": "Use POST for /reset and /step"}, 405)

    def log_message(self, format, *args):
        print(f"[MockEnvServer] {args[0]}")


def main():
    parser = argparse.ArgumentParser(description="Mock Environment Server for Testing")
    parser.add_argument("--port", type=int, default=8084, help="Port to run server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MockEnvHandler)
    print(f"Mock Environment Server running on http://{args.host}:{args.port}")
    print("Endpoints:")
    print("  POST /reset  - Reset environment, returns initial observation")
    print("  POST /step   - Execute action, returns (observation, reward, done, info)")
    print("  GET  /health - Health check")
    print("\nPress Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
