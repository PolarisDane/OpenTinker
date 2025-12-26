#!/usr/bin/env python3
import re
import requests
from typing import Any, Dict, Optional

from opentinker.environment.base_game import AbstractGame, StepResult


class CodeInterpreterMathGame(AbstractGame):
    """Multi-turn math game with internal code execution.

    This game is designed to work with GymEnvironmentInteraction and
    GenericAgentLoop (algorithm: agent_loop). Code execution happens
    in the step() method by calling an external sandbox server.

    The LLM should output:
    - Python code in ```python ... ``` blocks for execution
    - Final answer with \\boxed{...} format

    Attributes:
        sandbox_url: URL of the sandbox server (e.g., "http://localhost:8000/run_code")
        max_turns: Maximum number of turns before forcing termination
        code_pattern: Regex pattern to extract code blocks
    """

    # Reward constants
    REWARD_CORRECT = 1.0
    REWARD_INCORRECT = 0.0

    def __init__(
        self,
        sandbox_url: str = "http://localhost:8000/run_code",
        max_turns: int = 10,
        timeout: int = 30,
    ):
        """Initialize CodeInterpreterMathGame.

        Args:
            sandbox_url: URL of the sandbox server for code execution
            max_turns: Maximum conversation turns (0 = unlimited)
            timeout: Timeout for sandbox HTTP calls in seconds
        """
        self.sandbox_url = sandbox_url
        self.max_turns = max_turns
        self.timeout = timeout

        # Pattern to extract code blocks: ```python ... ``` or ```py ... ```
        self.code_pattern = re.compile(
            r"```(?:python|py)\s*(.*?)```", re.DOTALL | re.IGNORECASE
        )
        # Pattern to detect boxed answer
        self.boxed_pattern = re.compile(r"\\boxed\{([^}]+)\}")

        self._init_game_state()

    def _init_game_state(self):
        """Initialize/reset game state variables."""
        self.ground_truth = None
        self.data_source = "math"
        self.extra_info = {}
        self.turn_count = 0
        self.game_over = False
        self.execution_history = []  # Track code executions

    def reset(
        self,
        ground_truth: Optional[str] = None,
        data_source: str = "math",
        extra_info: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Reset the game with a new problem.

        Args:
            ground_truth: The correct answer (used for reward computation)
            data_source: Data source identifier for reward function
            extra_info: Additional info passed to reward function
            **kwargs: Ignored (for compatibility)

        Returns:
            Empty string (prompt already contains the problem)
        """
        self._init_game_state()
        self.ground_truth = ground_truth
        self.data_source = data_source
        self.extra_info = extra_info or {}

        # For multi-turn, no additional observation needed initially
        # The problem is already in the prompt from data generator
        return ""

    def step(self, action: str) -> StepResult:
        """Process the model's response.

        This method:
        1. Checks for final answer (\\boxed{})
        2. Extracts and executes Python code blocks
        3. Returns execution result as observation

        Args:
            action: The model's response (may contain code blocks or answer)

        Returns:
            StepResult with observation (code output) or reward (if done)
        """
        if self.game_over:
            return StepResult(
                observation="Game already over.",
                reward=0.0,
                done=True,
                info={"error": "game_over"},
            )

        self.turn_count += 1

        # Check for final answer first
        boxed_match = self.boxed_pattern.search(action)
        if boxed_match:
            self.game_over = True
            reward = self._compute_reward(action)

            # DEBUG: Log reward computation details
            print("[CodeInterpreterMathGame DEBUG] Final answer detected!")
            print(f"  - Turn count: {self.turn_count}")
            print(f"  - Code executions: {len(self.execution_history)}")
            print(f"  - Ground truth: {self.ground_truth}")
            print(f"  - Boxed answer: {boxed_match.group(1)}")
            print(f"  - Data source: {self.data_source}")
            print(f"  - Reward: {reward}")

            info = {
                "ground_truth": self.ground_truth,
                "data_source": self.data_source,
                "turn_count": self.turn_count,
                "solution_str": action,
                "has_final_answer": True,
                "execution_history": self.execution_history,
                "code_executions_count": len(
                    self.execution_history
                ),  # Track code execution count
            }
            info.update(self.extra_info)

            return StepResult(
                observation="",
                reward=reward,
                done=True,
                info=info,
            )

        # Extract code blocks
        code_blocks = self.code_pattern.findall(action)

        if code_blocks:
            # Execute all code blocks and collect results
            results = []
            for code in code_blocks:
                code = code.strip()
                if code:
                    output = self._execute_code(code)
                    results.append(output)
                    self.execution_history.append(
                        {
                            "turn": self.turn_count,
                            "code": code,
                            "output": output,
                        }
                    )

            # Combine all outputs
            observation = "\n".join(results) if results else "Code executed (no output)"

            # Check max turns
            if self.max_turns > 0 and self.turn_count >= self.max_turns:
                self.game_over = True
                # Try to compute reward from last response
                reward = self._compute_reward(action)

                return StepResult(
                    observation=f"{observation}\n\n[Maximum turns reached. Please provide final answer.]",
                    reward=reward,
                    done=True,
                    info={
                        "ground_truth": self.ground_truth,
                        "turn_count": self.turn_count,
                        "truncated": True,
                    },
                )

            # Continue conversation with execution result
            return StepResult(
                observation=observation,
                reward=0.0,
                done=False,
                info={"turn_count": self.turn_count},
            )

        # No code blocks and no boxed answer - prompt to continue
        if self.max_turns > 0 and self.turn_count >= self.max_turns:
            self.game_over = True
            reward = self._compute_reward(action)

            return StepResult(
                observation="Maximum turns reached.",
                reward=reward,
                done=True,
                info={
                    "ground_truth": self.ground_truth,
                    "turn_count": self.turn_count,
                    "truncated": True,
                },
            )

        # Encourage the model to provide code or answer
        return StepResult(
            observation="Please write Python code in ```python ... ``` blocks to solve the problem, or provide your final answer in \\boxed{} format.",
            reward=0.0,
            done=False,
            info={"turn_count": self.turn_count, "no_action": True},
        )

    def _execute_code(self, code: str) -> str:
        """Execute code via sandbox server.

        Args:
            code: Python code to execute

        Returns:
            Combined stdout and stderr from execution
        """
        try:
            response = requests.post(
                self.sandbox_url,
                json={"code": code},
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            run_result = result.get("run_result", {})
            stdout = run_result.get("stdout", "")
            stderr = run_result.get("stderr", "")

            output = stdout + stderr
            if not output.strip():
                output = "(Code executed successfully, no output)"

            return output.strip()

        except requests.exceptions.Timeout:
            return "[Error: Code execution timed out]"
        except requests.exceptions.ConnectionError:
            return "[Error: Cannot connect to sandbox server]"
        except Exception as e:
            return f"[Error executing code: {str(e)}]"

    def _compute_reward(self, solution_str: str) -> float:
        """Compute reward by comparing solution to ground truth.

        Uses the default_compute_score from verl for consistency.
        """
        try:
            from verl.utils.reward_score import default_compute_score

            score = default_compute_score(
                data_source=self.data_source,
                solution_str=solution_str,
                ground_truth=self.ground_truth,
                extra_info=self.extra_info,
            )

            # Handle both dict and scalar return values
            if isinstance(score, dict):
                return float(score.get("score", 0.0))
            return float(score)

        except Exception as e:
            # Fallback: simple string matching
            import logging

            logging.warning(f"Reward computation failed, using fallback: {e}")
            return self._simple_match(solution_str)

    def _simple_match(self, solution_str: str) -> float:
        """Simple fallback: check if ground_truth appears in solution."""
        if self.ground_truth is None:
            return 0.0

        # Extract boxed answer if present
        boxed_match = self.boxed_pattern.search(solution_str)
        if boxed_match:
            solution_str = boxed_match.group(1)

        # Normalize and compare
        solution_normalized = solution_str.strip().lower()
        gt_normalized = str(self.ground_truth).strip().lower()

        return (
            self.REWARD_CORRECT
            if gt_normalized in solution_normalized
            else self.REWARD_INCORRECT
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for math problems with code interpreter.

        This prompt instructs the LLM to use Python code for solving problems.
        """
        return (
            "You are a helpful assistant that solves math problems using Python code.\n\n"
            "When solving problems:\n"
            "1. Think through the problem step by step\n"
            "2. Write Python code in ```python ... ``` blocks to perform calculations\n"
            "3. You can use libraries like sympy, numpy, scipy for mathematical computations\n"
            "4. After getting the result from code execution, provide your final answer in \\boxed{} format\n\n"
            "Example:\n"
            "```python\n"
            "import sympy\n"
            "result = sympy.simplify(sympy.sqrt(8))\n"
            "print(result)\n"
            "```\n\n"
            "After seeing the output, write: The answer is \\boxed{2*sqrt(2)}"
        )

    def get_initial_user_message(self) -> str:
        """Return the initial user message."""
        return "Please solve the following problem:"

    def get_state(self) -> Dict[str, Any]:
        """Return current game state."""
        return {
            "ground_truth": self.ground_truth,
            "data_source": self.data_source,
            "turn_count": self.turn_count,
            "game_over": self.game_over,
            "execution_history": self.execution_history,
        }

    def get_interaction_name(self) -> str:
        """Return interaction name for math code interpreter."""
        return "math_code_interpreter"

    # =========================================================================
    # Data Generation Methods (for training with static dataset)
    # =========================================================================

    def generate_initial_state(self) -> Dict[str, Any]:
        """For static dataset, this is not used - data comes from generator."""
        return {}

    def get_user_message_with_state(self, **kwargs) -> str:
        """Generate user message - for static data, prompt comes from dataset."""
        return self.get_initial_user_message()
