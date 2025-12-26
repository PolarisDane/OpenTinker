"""Gomoku Environment Module for LLM Training.

Usage:
    from opentinker.environment.base_game_environment import GameEnvironment
    from opentinker.environment.gomoku import GomokuGame
    from opentinker.environment.game_stats_client import GameStatsClient

    env = GameEnvironment(game_class=GomokuGame, config=config)
    stats_client = GameStatsClient(env_endpoint)

    # Optional: GomokuGameStats for server-side metrics
    from opentinker.environment.gomoku import GomokuGameStats  # may be None
"""

from .gomoku_game import GomokuGame

# GomokuGameStats is optional - only available if gomoku_stats.py exists
try:
    from .gomoku_stats import GomokuGameStats
except ImportError:
    GomokuGameStats = None

__all__ = [
    "GomokuGame",
    "GomokuGameStats",
]
