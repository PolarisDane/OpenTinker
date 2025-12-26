#!/usr/bin/env python3
"""Math Environment Server - HTTP server for math problem solving.

This script starts a math game server using the generic base_game_server.
For single-turn math problems, the server:
- Receives reset() with ground_truth
- Receives step() with model's answer
- Returns reward computed by MathGame

Usage:
    python math_server.py
    # Or with custom port:
    python math_server.py --port 8082

    # For multi-worker mode (faster handling of concurrent requests):
    uvicorn opentinker.environment.math.math_server:app --host 0.0.0.0 --port 8082 --workers 4
"""

import argparse
from opentinker.environment.base_game_server import run_game_server
from opentinker.environment.math.math_game import MathGame

# Pre-import reward function to avoid first-request latency
# (The first import of verl.utils.reward_score can be slow)
try:
    from verl.utils.reward_score import default_compute_score
except ImportError:
    pass

# # Module-level app for uvicorn multi-worker mode
# # Usage: uvicorn opentinker.environment.math.math_server:app --host 0.0.0.0 --port 8082 --workers 4
# app = create_game_app(game_class=MathGame)


def main():
    parser = argparse.ArgumentParser(description="Math Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8082, help="Server port")
    parser.add_argument(
        "--max_retries", type=int, default=0, help="Max retry attempts (0=single turn)"
    )
    args = parser.parse_args()

    print("\nMath Game Server Configuration:")
    print(f"  Single-turn mode: {'Yes' if args.max_retries == 0 else 'No'}")
    print(f"  Max retries: {args.max_retries}")
    print("\nReward structure:")
    print(f"  Correct answer: +{MathGame.REWARD_CORRECT}")
    print(f"  Incorrect answer: {MathGame.REWARD_INCORRECT}")

    run_game_server(
        game_class=MathGame,
        host=args.host,
        port=args.port,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
