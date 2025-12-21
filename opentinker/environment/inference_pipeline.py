#!/usr/bin/env python3
"""
Generic Inference Pipeline for OpenTinker Environments

This module provides a reusable inference pipeline that can be used with any
AbstractGame implementation. Environment-specific scripts just need to configure
the game class and data paths.

Usage:
    from opentinker.environment.inference_pipeline import (
        InferencePipeline, run_inference
    )
    from opentinker.environment.math import MathGame
    
    # Simple usage
    results = run_inference(
        model_path="/path/to/checkpoint",
        data_path="/path/to/test.jsonl",
        game_class=MathGame,
        env_endpoint="http://localhost:8088",
    )
    
    # Or with more control
    pipeline = InferencePipeline(
        model_path="/path/to/checkpoint",
        env_endpoint="http://localhost:8088",
    )
    result = await pipeline.run_single_inference(messages, env_kwargs)
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

import aiohttp
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from opentinker.environment.base_game import AbstractGame
from opentinker.environment.static_data_generator import StaticDatasetGenerator


@dataclass
class InferenceResult:
    """Result of a single inference trajectory."""
    sample_id: str
    messages: List[Dict[str, Any]]
    prompt_text: str
    response_text: str
    reward: float
    done: bool
    num_turns: int
    ground_truth: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_id": self.sample_id,
            "prompt_text": self.prompt_text,
            "response_text": self.response_text,
            "reward": self.reward,
            "done": self.done,
            "num_turns": self.num_turns,
            "ground_truth": self.ground_truth,
            "info": self.info,
            "messages": self.messages,
        }


class RemoteEnvironmentClient:
    """Async HTTP client for remote game server communication.
    
    Supports multi-job statistics isolation via job_id parameter.
    """
    
    def __init__(self, env_endpoint: str, job_id: str = "default", timeout: float = 60.0):
        """Initialize environment client.
        
        Args:
            env_endpoint: URL of the game server (e.g., http://localhost:8082)
            job_id: Job identifier for statistics isolation (default: "default")
            timeout: Request timeout in seconds
        """
        self.env_endpoint = env_endpoint
        self.job_id = job_id
        self.timeout = aiohttp.ClientTimeout(total=timeout)
    
    async def reset(self, instance_id: str, **kwargs) -> Dict[str, Any]:
        """Call env.reset() on remote server."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.env_endpoint}/reset",
                json={"instance_id": instance_id, "job_id": self.job_id, **kwargs},
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Environment reset failed: {await response.text()}")
                return await response.json()
    
    async def step(self, instance_id: str, action: str) -> Dict[str, Any]:
        """Call env.step(action) on remote server."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.env_endpoint}/step",
                json={"instance_id": instance_id, "job_id": self.job_id, "action": action},
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Environment step failed: {await response.text()}")
                return await response.json()
    
    async def health_check(self) -> bool:
        """Check if the server is running."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.env_endpoint}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
        except Exception:
            return False


class InferencePipeline:
    """Generic inference pipeline for OpenTinker environments.
    
    Supports two modes:
    1. Offline mode: Load model locally with vLLM (model_path required)
    2. Server mode: Connect to vLLM server API (vllm_server_url required)
    
    Args:
        model_path: Path to HuggingFace model checkpoint (for offline mode)
        tokenizer_path: Path to tokenizer (defaults to model_path)
        vllm_server_url: vLLM server URL for server mode (e.g., "http://localhost:8000")
        env_endpoint: Remote game server URL
        max_user_turns: Max environment interaction turns (0 = single-turn)
        max_assistant_turns: Max model response turns
        tensor_parallel_size: Number of GPUs for tensor parallelism (offline mode only)
        gpu_memory_utilization: GPU memory fraction to use (offline mode only)
        trust_remote_code: Whether to trust remote code in model
        
    Examples:
        # Offline mode (load model locally)
        pipeline = InferencePipeline(model_path="/path/to/model")
        
        # Server mode (connect to vLLM server)
        pipeline = InferencePipeline(
            vllm_server_url="http://localhost:8000",
            tokenizer_path="/path/to/tokenizer",  # Required for chat template
        )
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        vllm_server_url: Optional[str] = None,
        env_endpoint: str = "http://localhost:8088",
        job_id: str = "default",
        max_user_turns: int = 0,
        max_assistant_turns: int = 1,
        max_context_length: int = 30000,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
    ):
        self.env_endpoint = env_endpoint
        self.job_id = job_id
        self.max_user_turns = max_user_turns
        self.max_assistant_turns = max_assistant_turns
        self.max_context_length = max_context_length
        self.vllm_server_url = vllm_server_url
        
        # Determine mode
        if vllm_server_url:
            self.mode = "server"
            self.model = None
            tokenizer_path = tokenizer_path or model_path
            if not tokenizer_path:
                raise ValueError("tokenizer_path is required for server mode")
        elif model_path:
            self.mode = "offline"
        else:
            raise ValueError("Either model_path or vllm_server_url is required")
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        
        # Load tokenizer
        print(f"Loading tokenizer from {self.tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model for offline mode
        if self.mode == "offline":
            print(f"Loading model from {self.model_path} with vLLM (offline mode)...")
            self.model = LLM(
                model=self.model_path,
                tokenizer=self.tokenizer_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
            )
        else:
            print(f"Using vLLM server at {self.vllm_server_url} (server mode)...")
        
        self.env_client = RemoteEnvironmentClient(env_endpoint, job_id=self.job_id)
        print(f"✓ Inference pipeline initialized ({self.mode} mode, job_id={self.job_id})")
    
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response (offline mode only, use generate_response_async for server mode)."""
        if self.mode == "server":
            # For synchronous call in server mode, use asyncio
            return asyncio.get_event_loop().run_until_complete(
                self.generate_response_async(messages, temperature, top_p, max_tokens)
            )
        
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        outputs = self.model.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
    
    async def generate_response_async(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response asynchronously (works for both modes)."""
        if self.mode == "offline":
            # Run offline generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self.generate_response(messages, temperature, top_p, max_tokens)
            )
        
        # Server mode: use OpenAI-compatible API
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.vllm_server_url}/v1/completions",
                json={
                    "model": self.tokenizer_path,  # Use model/tokenizer path as model name
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                timeout=aiohttp.ClientTimeout(total=300)  # 5 min timeout for long generations
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"vLLM server error: {error_text}")
                data = await response.json()
                return data["choices"][0]["text"]
    
    async def generate_batch_async(
        self,
        prompts: List[str],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
    ) -> List[str]:
        """Generate responses for a batch of prompts (server mode optimized)."""
        if self.mode == "offline":
            # Offline batch generation
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            outputs = self.model.generate(prompts, sampling_params)
            return [out.outputs[0].text for out in outputs]
        
        # Server mode: parallel requests
        async def generate_one(prompt):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.vllm_server_url}/v1/completions",
                    json={
                        "model": self.tokenizer_path,  # Use model/tokenizer path as model name
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                    },
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise RuntimeError(f"vLLM server error: {error_text}")
                    data = await response.json()
                    return data["choices"][0]["text"]
        
        # Run all requests in parallel
        tasks = [generate_one(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    async def run_single_inference(
        self,
        messages: List[Dict[str, Any]],
        env_kwargs: Dict[str, Any],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        max_tokens_per_turn: Optional[int] = None,
    ) -> InferenceResult:
        """Run inference for a single sample with environment interaction.
        
        Args:
            messages: Initial conversation messages
            env_kwargs: Environment kwargs (ground_truth, etc.)
            temperature: Sampling temperature
            top_p: Nucleus sampling
            max_tokens: Total max tokens for generation (fallback)
            max_tokens_per_turn: Per-turn token limit (overrides max_tokens if set)
        """
        from uuid import uuid4
        
        instance_id = uuid4().hex
        conversation = list(messages)
        
        # Reset environment
        await self.env_client.reset(instance_id, **env_kwargs)
        
        user_turns = 0
        assistant_turns = 0
        cumulative_reward = 0.0
        done = False
        info_list = []
        total_response_tokens = 0  # Track total tokens generated across all turns
        
        while not done:
            if self.max_assistant_turns and assistant_turns >= self.max_assistant_turns:
                break
            
            # Check if we've exhausted total response budget
            if total_response_tokens >= max_tokens:
                print(f"⚠ Total response tokens ({total_response_tokens}) reached limit ({max_tokens}), ending")
                break
            
            # Check context length before generation
            prompt = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            prompt_tokens = len(self.tokenizer.encode(prompt))
            if prompt_tokens > self.max_context_length:
                print(f"⚠ Context length ({prompt_tokens}) exceeds limit ({self.max_context_length}), ending game")
                break
            
            # Calculate remaining tokens for this turn
            remaining_tokens = max_tokens - total_response_tokens
            # Use per-turn limit if set, otherwise use remaining budget
            turn_max_tokens = min(
                max_tokens_per_turn if max_tokens_per_turn else remaining_tokens,
                remaining_tokens
            )
            
            response = await self.generate_response_async(
                conversation, temperature=temperature, top_p=top_p, max_tokens=turn_max_tokens
            )
            
            # Track tokens generated this turn
            response_tokens = len(self.tokenizer.encode(response))
            total_response_tokens += response_tokens
            
            assistant_turns += 1
            conversation.append({"role": "assistant", "content": response})
            
            if self.max_user_turns and user_turns >= self.max_user_turns:
                if self.max_user_turns == 0:
                    step_result = await self.env_client.step(instance_id, response)
                    cumulative_reward += step_result.get("reward", 0.0)
                    done = step_result.get("done", True)
                    info_list.append(step_result.get("info", {}))
                break
            
            step_result = await self.env_client.step(instance_id, response)
            observation = step_result.get("observation", "")
            reward = step_result.get("reward", 0.0)
            done = step_result.get("done", False)
            
            cumulative_reward += reward
            info_list.append(step_result.get("info", {}))
            user_turns += 1
            
            if done:
                break
            
            if observation and observation.strip():
                conversation.append({"role": "user", "content": observation})
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response_text = "\n".join(m["content"] for m in conversation if m["role"] == "assistant")
        
        return InferenceResult(
            sample_id=instance_id,
            messages=conversation,
            prompt_text=prompt_text,
            response_text=response_text,
            reward=cumulative_reward,
            done=done,
            num_turns=user_turns + assistant_turns,
            ground_truth=env_kwargs.get("ground_truth"),
            info={"env_info": info_list}
        )
    
    async def run_batch_inference(
        self,
        samples: List[Dict[str, Any]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 4096,
        max_tokens_per_turn: Optional[int] = None,
        output_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[InferenceResult]:
        """Run inference on a batch of samples."""
        results = []
        iterator = tqdm(samples, desc="Inference") if show_progress else samples
        
        for sample in iterator:
            result = await self.run_single_inference(
                messages=sample["prompt"],
                env_kwargs=sample.get("env_kwargs", {}),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_tokens_per_turn=max_tokens_per_turn,
            )
            results.append(result)
            
            if output_path:
                with open(output_path, "a") as f:
                    f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        
        return results


def load_samples(
    data_path: Optional[str],
    game_class: Type[AbstractGame],
    max_samples: Optional[int] = None,
    **game_kwargs,
) -> List[Dict[str, Any]]:
    """Load samples from data file OR generate dynamically.
    
    For static datasets (Math, etc.): Pass data_path to load from file.
    For dynamic environments (Gomoku, etc.): Pass data_path=None to generate.
    
    Args:
        data_path: Path to data file (parquet/jsonl), or None for dynamic generation
        game_class: AbstractGame subclass
        max_samples: Number of samples (required if data_path is None)
        **game_kwargs: Arguments for game class constructor
        
    Returns:
        List of sample dicts with 'prompt' and 'env_kwargs'
    """
    from opentinker.environment.base_game import GameDataGenerator
    
    if data_path is not None:
        # Static dataset - load from file
        game = game_class(**game_kwargs)
        system_prompt = game.get_system_prompt()
        
        generator = StaticDatasetGenerator(
            data_paths=[data_path],
            interaction_name=game.get_interaction_name(),
            prompt_key="prompt",
            ground_truth_key="ground_truth",
            shuffle=False,
            system_prompt=system_prompt,
        )
        
        num_samples = len(generator) if max_samples is None else min(max_samples, len(generator))
        samples = [generator.generate_sample(i) for i in range(num_samples)]
        print(f"Loaded {len(samples)} samples from {data_path}")
    else:
        # Dynamic generation - use GameDataGenerator
        if max_samples is None:
            raise ValueError("max_samples is required when data_path is None (dynamic generation)")
        
        generator = GameDataGenerator(
            game_class=game_class,
            game_kwargs=game_kwargs,
            seed=42,  # For reproducibility
        )
        
        samples = [generator.generate_sample(i) for i in range(max_samples)]
        print(f"Generated {len(samples)} samples dynamically using {game_class.__name__}")
    
    return samples


def generate_samples(
    game_class: Type[AbstractGame],
    num_samples: int,
    seed: int = 42,
    **game_kwargs,
) -> List[Dict[str, Any]]:
    """Generate samples dynamically using a game's generate_initial_state().
    
    This is useful for environments like Gomoku that don't use static data files.
    
    Args:
        game_class: AbstractGame subclass (e.g., GomokuGame)
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        **game_kwargs: Arguments for game class constructor
        
    Returns:
        List of sample dicts with 'prompt' and 'env_kwargs'
    """
    from opentinker.environment.base_game import GameDataGenerator
    
    generator = GameDataGenerator(
        game_class=game_class,
        game_kwargs=game_kwargs,
        seed=seed,
    )
    
    samples = [generator.generate_sample(i) for i in range(num_samples)]
    print(f"Generated {len(samples)} samples using {game_class.__name__}")
    return samples


def run_inference(
    model_path: Optional[str] = None,
    data_path: Optional[str] = None,
    game_class: Type[AbstractGame] = None,
    env_endpoint: str = "http://localhost:8088",
    job_id: str = "default",
    vllm_server_url: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    output_path: Optional[str] = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    max_tokens_per_turn: Optional[int] = None,
    max_samples: Optional[int] = None,
    max_user_turns: int = 0,
    max_assistant_turns: int = 1,
    max_context_length: int = 30000,
    tensor_parallel_size: int = 1,
    **game_kwargs,
) -> List[InferenceResult]:
    """Convenience function to run inference with minimal configuration.
    
    Supports two modes:
    1. Offline mode: Load model locally (model_path required)
    2. Server mode: Use vLLM server (vllm_server_url required)
    
    Args:
        model_path: Path to HuggingFace checkpoint (offline mode)
        data_path: Path to test data file, or None for dynamic generation
        game_class: AbstractGame subclass (e.g., MathGame, GomokuGame)
        env_endpoint: Remote game server URL
        vllm_server_url: vLLM server URL (server mode, e.g., "http://localhost:8000")
        tokenizer_path: Tokenizer path (required for server mode if model_path not set)
        output_path: Optional output file for results
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling parameter
        max_tokens: Max generation tokens
        max_samples: Number of samples (required if data_path is None)
        max_user_turns: Max environment turns (0 = single-turn)
        max_assistant_turns: Max model turns
        max_context_length: Max context length before truncation (default 30000)
        tensor_parallel_size: GPU parallelism (offline mode only)
        **game_kwargs: Arguments for game class constructor
        
    Returns:
        List of InferenceResult objects
        
    Examples:
        # Offline mode (load model locally)
        run_inference(model_path="/path/to/model", data_path="test.jsonl", game_class=MathGame)
        
        # Server mode (connect to vLLM server)
        run_inference(
            vllm_server_url="http://localhost:8000",
            tokenizer_path="/path/to/tokenizer",
            data_path="test.jsonl", 
            game_class=MathGame,
        )
        
        # Dynamic generation (Gomoku)
        run_inference(model_path, data_path=None, game_class=GomokuGame, max_samples=10)
    """
    if game_class is None:
        raise ValueError("game_class is required")
    
    # Load samples
    samples = load_samples(data_path, game_class, max_samples, **game_kwargs)
    
    # Initialize pipeline
    pipeline = InferencePipeline(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        vllm_server_url=vllm_server_url,
        env_endpoint=env_endpoint,
        job_id=job_id,
        max_user_turns=max_user_turns,
        max_assistant_turns=max_assistant_turns,
        max_context_length=max_context_length,
        tensor_parallel_size=tensor_parallel_size,
    )
    
    # Check server health
    async def check_and_run():
        healthy = await pipeline.env_client.health_check()
        if not healthy:
            raise RuntimeError(f"Game server not available at {env_endpoint}")
        print(f"✓ Connected to game server at {env_endpoint}")
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            open(output_path, "w").close()
        
        return await pipeline.run_batch_inference(
            samples=samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_tokens_per_turn=max_tokens_per_turn,
            output_path=output_path,
        )
    
    start_time = time.time()
    results = asyncio.run(check_and_run())
    elapsed = time.time() - start_time
    
    # Print summary
    rewards = [r.reward for r in results]
    print(f"\n{'='*50}")
    print(f"Inference Complete: {len(results)} samples in {elapsed:.1f}s")
    print(f"Mean reward: {sum(rewards)/len(rewards):.4f}")
    correct = sum(1 for r in rewards if r > 0.5)
    print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
    
    return results