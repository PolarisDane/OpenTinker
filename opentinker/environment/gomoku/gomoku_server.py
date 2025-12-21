#!/usr/bin/env python3
"""Gomoku Environment Server - Simplified launcher.

This script starts a Gomoku game server using the generic base_game_server.

Usage:
    python gomoku_server.py
    # Or with custom config:
    python gomoku_server.py --port 8081 --board_size 9
"""

import argparse
from opentinker.environment.base_game_server import run_game_server
from opentinker.environment.gomoku.gomoku_game import GomokuGame

# GomokuGameStats is optional - falls back to BaseGameStats if not available
try:
    from opentinker.environment.gomoku.gomoku_stats import GomokuGameStats
except ImportError:
    GomokuGameStats = None


def main():
    parser = argparse.ArgumentParser(description="Gomoku Game Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8081, help="Server port")
    parser.add_argument("--board_size", type=int, default=9, help="Board size")
    parser.add_argument("--max_total_steps", type=int, default=40, help="Max steps")
    args = parser.parse_args()
    
    print(f"\nGomoku Game Configuration:")
    print(f"  Board size: {args.board_size}x{args.board_size}")
    print(f"  Max steps: {args.max_total_steps}")
    print(f"\nReward structure:")
    print(f"  Win: +{GomokuGame.REWARD_WIN}")
    print(f"  Loss: {GomokuGame.REWARD_LOSS}")
    print(f"  Invalid format: {GomokuGame.REWARD_INVALID_FORMAT}")
    print(f"  Timeout: {GomokuGame.REWARD_TIMEOUT}")
    
    if GomokuGameStats:
        print(f"\nUsing GomokuGameStats for win/loss/draw tracking")
    else:
        print(f"\nUsing BaseGameStats (GomokuGameStats not available)")
    
    run_game_server(
        game_class=GomokuGame,
        host=args.host,
        port=args.port,
        stats_class=GomokuGameStats,  # None falls back to BaseGameStats
        board_size=args.board_size,
        max_total_steps=args.max_total_steps,
    )


if __name__ == "__main__":
    main()
