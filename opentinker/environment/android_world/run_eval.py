#!/usr/bin/env python3
"""Android World Evaluation Runner.

Runs evaluation on specified splits (test_id, test_ood, or both) and
produces comprehensive metrics reports.

Usage:
    # Evaluate OOD test set
    python run_eval.py --split test_ood

    # Evaluate both ID and OOD
    python run_eval.py --split test_id test_ood

    # Custom config and output
    python run_eval.py --config path/to/task_sets.yaml --output_dir ./eval_results

    # Override instances per task
    python run_eval.py --split test_ood --n_instances 5

    # Evaluate a specific checkpoint (local model)
    python run_eval.py --split test_ood --model_path /path/to/checkpoint

    # Evaluate a checkpoint with a separate tokenizer
    python run_eval.py --split test_ood --model_path /path/to/checkpoint --tokenizer_path Qwen/Qwen2.5-3B-Instruct

    # Evaluate using a running vLLM server
    python run_eval.py --split test_ood --vllm_server_url http://localhost:8000 --tokenizer_path Qwen/Qwen2.5-3B-Instruct

    # Validate config against registry (no episodes run)
    python run_eval.py --validate_only

    # Compute metrics from previously saved eval JSON
    python run_eval.py --from_json eval_results/eval_test_ood_1708000000.json
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

# Add project root to path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opentinker.environment.android_world.task_set_config import TaskSetConfig
from opentinker.environment.android_world.evaluator import (
    AndroidWorldEvaluator,
    EvalResults,
)

logger = logging.getLogger(__name__)


def build_agent_fn(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    vllm_server_url: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> Callable[[str], str]:
    """Build an agent function from a model checkpoint or vLLM server.

    The returned callable takes an observation string (the formatted prompt
    from AndroidWorldGame) and returns an action string in the format:
        Reason: ...
        Action: {"action_type": ...}

    Args:
        model_path: Path to a HuggingFace model checkpoint directory.
                    Can be a training checkpoint (e.g., ckpt/step_100/) or
                    a base model (e.g., Qwen/Qwen2.5-3B-Instruct).
        tokenizer_path: Path to tokenizer. Defaults to model_path.
        vllm_server_url: URL of a running vLLM server (e.g., http://localhost:8000).
                         If provided, model_path is not needed (only tokenizer_path).
        tensor_parallel_size: Number of GPUs for tensor parallelism (offline mode).
        gpu_memory_utilization: GPU memory fraction (offline mode).
        temperature: Sampling temperature (0.0 = greedy).
        max_tokens: Maximum tokens to generate per action.

    Returns:
        A callable agent_fn(observation: str) -> str.
    """
    from transformers import AutoTokenizer

    # Determine mode and load resources
    tok_path = tokenizer_path or model_path
    if not tok_path:
        raise ValueError(
            "At least one of --model_path or --tokenizer_path is required "
            "to build an agent."
        )

    print(f"Loading tokenizer from {tok_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if vllm_server_url:
        # ---------- Server mode ----------
        import asyncio
        import aiohttp

        print(f"Using vLLM server at {vllm_server_url} (server mode)")

        # Resolve or create an event loop for synchronous calls
        def _generate_server(prompt: str) -> str:
            async def _post():
                payload = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 1.0,
                }
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{vllm_server_url}/v1/completions", json=payload
                    ) as resp:
                        data = await resp.json()
                        return data["choices"][0]["text"]

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    return pool.submit(asyncio.run, _post()).result()
            else:
                return asyncio.run(_post())

        generate = _generate_server

    elif model_path:
        # ---------- Offline mode ----------
        from vllm import LLM, SamplingParams

        print(
            f"Loading model from {model_path} with vLLM "
            f"(tp={tensor_parallel_size}, gpu_mem={gpu_memory_utilization})..."
        )
        model = LLM(
            model=model_path,
            tokenizer=tok_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
        )

        def _generate_offline(prompt: str) -> str:
            outputs = model.generate([prompt], sampling_params)
            return outputs[0].outputs[0].text

        generate = _generate_offline
    else:
        raise ValueError(
            "Either --model_path or --vllm_server_url is required "
            "to build an agent."
        )

    print("Agent function ready.")

    # Build the agent closure
    def agent_fn(observation: str) -> str:
        """Generate an action given the formatted observation prompt."""
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant capable of operating an Android device."},
            {"role": "user", "content": observation},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return generate(prompt)

    return agent_fn


def validate_config(config: TaskSetConfig):
    """Validate config and print summary."""
    print(config.summary())
    print()

    # Try to validate against registry
    try:
        from android_world import registry
        task_registry = registry.TaskRegistry()
        all_tasks = task_registry.get_registry(registry.TaskRegistry.ANDROID_WORLD_FAMILY)
        report = config.validate_against_registry(all_tasks)
        all_valid = True
        for split, invalid in report.items():
            if invalid:
                all_valid = False
                print(f"  ⚠ [{split}] Invalid tasks: {invalid}")
        if all_valid:
            print("  ✓ All tasks valid against AndroidWorld registry.")
    except ImportError:
        print("  ⚠ android_world not installed — cannot validate task names against registry.")


def run_eval(
    splits: List[str],
    config: TaskSetConfig,
    n_instances: Optional[int] = None,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: str = "./eval_results",
    emulator_console_port: Optional[int] = None,
    emulator_grpc_port: Optional[int] = None,
    mock: bool = False,
    agent_fn=None,
):
    """Run evaluation on specified splits."""
    from opentinker.environment.android_world.android_world_game import AndroidWorldGame

    all_results = {}

    for split in splits:
        print(f"\n{'=' * 70}")
        print(f"  Starting evaluation: {split.upper()}")
        print(f"{'=' * 70}")

        tasks = config.get_tasks(split)
        n = n_instances or config.n_instances_per_task
        print(f"  Tasks: {len(tasks)} types × {n} instances = {len(tasks) * n} episodes")
        print(f"  Max steps/episode: {max_steps or config.eval_max_steps}")
        print()

        # Create game instance
        game = AndroidWorldGame(
            max_steps=max_steps or config.eval_max_steps,
            task_types=tasks,
            split=split,
            emulator_console_port=emulator_console_port,
            emulator_grpc_port=emulator_grpc_port,
        )

        evaluator = AndroidWorldEvaluator(
            game=game,
            task_set_config=config,
            split=split,
            agent_fn=agent_fn,
            n_instances_per_task=n,
            max_steps=max_steps,
            seed=seed,
            output_dir=output_dir,
            verbose=True,
        )

        results = evaluator.run()
        all_results[split] = results

    # Print comparison if multiple splits
    if len(all_results) > 1:
        print_comparison(all_results)

    return all_results


def print_comparison(all_results: dict):
    """Print a side-by-side comparison of multiple splits."""
    print(f"\n{'=' * 70}")
    print("  CROSS-SPLIT COMPARISON")
    print(f"{'=' * 70}")
    print(
        f"  {'Metric':<30s}" + "".join(f"{s:>15s}" for s in all_results.keys())
    )
    print("  " + "-" * (30 + 15 * len(all_results)))

    metrics = [
        ("Success Rate", lambda r: f"{r.success_rate:.1%}"),
        ("Avg Steps (all)", lambda r: f"{r.avg_steps:.2f}"),
        ("Avg Steps (success)", lambda r: f"{r.avg_steps_on_success:.2f}"),
        ("Median Steps (success)", lambda r: f"{r.median_steps_on_success:.1f}"),
        ("Avg Reward", lambda r: f"{r.avg_reward:.3f}"),
        ("Timeout Rate", lambda r: f"{r.timeout_rate:.1%}"),
        ("Avg Invalid Actions", lambda r: f"{r.avg_invalid_actions:.2f}"),
        ("Error Rate", lambda r: f"{r.error_rate:.1%}"),
        ("Total Episodes", lambda r: f"{r.n_episodes}"),
        ("Unique Tasks", lambda r: f"{r.n_tasks}"),
        ("Wall Time (s)", lambda r: f"{r.total_wall_time:.1f}"),
    ]

    for name, fn in metrics:
        vals = "".join(f"{fn(r):>15s}" for r in all_results.values())
        print(f"  {name:<30s}{vals}")

    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Android World Evaluation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--split",
        nargs="+",
        default=["test_id", "test_ood"],
        choices=["train", "test_id", "test_ood"],
        help="Split(s) to evaluate (default: test_id test_ood)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to task_sets.yaml (default: built-in config)",
    )
    parser.add_argument(
        "--n_instances",
        type=int,
        default=None,
        help="Number of instances per task (overrides config)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per episode (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save results JSON",
    )
    parser.add_argument(
        "--emulator_console_port",
        type=int,
        default=None,
        help="Emulator console port",
    )
    parser.add_argument(
        "--emulator_grpc_port",
        type=int,
        default=None,
        help="Emulator gRPC port",
    )

    # --- Checkpoint / Model arguments ---
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=(
            "Path to a HuggingFace model checkpoint to evaluate. "
            "Can be a training checkpoint dir (e.g., ckpt/step_100/) or "
            "a model name (e.g., Qwen/Qwen2.5-3B-Instruct). "
            "If omitted and --vllm_server_url is not set, a dummy agent is used."
        ),
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to --model_path)",
    )
    parser.add_argument(
        "--vllm_server_url",
        type=str,
        default=None,
        help=(
            "URL of a running vLLM server (e.g., http://localhost:8000). "
            "When set, the model is served remotely and --model_path is not needed "
            "(but --tokenizer_path is required)."
        ),
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (offline mode, default: 1)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory fraction to use (offline mode, default: 0.9)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy, default: 0.0)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per action (default: 4096)",
    )

    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate config, don't run episodes",
    )
    parser.add_argument(
        "--from_json",
        type=str,
        default=None,
        help="Recompute metrics from a previously saved eval JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Mode 1: Recompute from JSON
    if args.from_json:
        results = EvalResults.from_json(args.from_json)
        print(results.summary())
        return

    # Load config
    config = TaskSetConfig(args.config)

    # Mode 2: Validate only
    if args.validate_only:
        validate_config(config)
        return

    # Mode 3: Run evaluation
    validate_config(config)

    # Build agent function from checkpoint / vLLM server if specified
    agent_fn = None
    if args.model_path or args.vllm_server_url:
        agent_fn = build_agent_fn(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            vllm_server_url=args.vllm_server_url,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    else:
        print(
            "\n  NOTE: No --model_path or --vllm_server_url specified.\n"
            "        Using dummy agent (always declares 'complete').\n"
            "        To evaluate a checkpoint, add --model_path <path>.\n"
        )

    run_eval(
        splits=args.split,
        config=config,
        n_instances=args.n_instances,
        max_steps=args.max_steps,
        seed=args.seed if args.seed is not None else config.eval_seed,
        output_dir=args.output_dir,
        emulator_console_port=args.emulator_console_port,
        emulator_grpc_port=args.emulator_grpc_port,
        agent_fn=agent_fn,
    )


if __name__ == "__main__":
    main()
