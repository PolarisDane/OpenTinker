"""Math Environment Package.

Provides MathGame and related components for math problem solving:
- MathGame: Single-turn math problem solving with rewards computed in step()
- CodeInterpreterMathGame: Multi-turn math with code interpreter tool support
"""

from opentinker.environment.math.math_game import MathGame
from opentinker.environment.math.math_tool_game import CodeInterpreterMathGame

__all__ = ["MathGame", "CodeInterpreterMathGame"]
