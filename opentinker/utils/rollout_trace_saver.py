#!/usr/bin/env python3
"""Rollout Trace Saver for Algorithm Verification.

This module provides utilities to save rollout traces to JSON files
and initialize Weave (W&B) tracing for debugging and verification.

Usage:
    from opentinker.utils.rollout_trace_saver import (
        RolloutTraceSaver,
        init_weave_tracing,
    )
    
    # Initialize Weave (optional)
    init_weave_tracing(project_name="opentinker/generic-env", experiment_name="run_001")
    
    # Create saver
    saver = RolloutTraceSaver(output_dir="/path/to/traces")
    
    # Save a trace
    saver.save_trace(
        sample_id="sample_001",
        messages=[...],
        response_ids=[...],
        response_mask=[...],
        reward=1.0,
        turn_scores=[0.1, 0.2, 0.7],
    )
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RolloutTrace:
    """A single rollout trace for verification."""
    
    # Identifiers
    sample_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    global_step: Optional[int] = None
    
    # Input/Output
    initial_prompt: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tokenized data
    prompt_ids: Optional[List[int]] = None
    response_ids: Optional[List[int]] = None
    response_mask: Optional[List[int]] = None
    
    # Decoded text (if available)
    prompt_text: Optional[str] = None
    response_text: Optional[str] = None
    
    # Reward and scoring
    reward_score: Optional[float] = None
    turn_scores: List[float] = field(default_factory=list)
    env_info: List[Dict[str, Any]] = field(default_factory=list)
    
    # Turn tracking
    num_user_turns: int = 0
    num_assistant_turns: int = 0
    total_turns: int = 0
    
    # Extra metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class RolloutTraceSaver:
    """Saves rollout traces to JSON files for verification.
    
    Features:
    - Saves individual traces as JSON files
    - Aggregates traces into batch files
    - Supports streaming mode (append to single file)
    - Thread-safe for concurrent writes
    """
    
    def __init__(
        self,
        output_dir: str = "/tmp/rollout_traces",
        save_individual: bool = True,
        save_batch: bool = True,
        batch_size: int = 100,
        streaming_mode: bool = False,
    ):
        """Initialize the trace saver.
        
        Args:
            output_dir: Directory to save traces
            save_individual: Save each trace as a separate file
            save_batch: Save traces in batches
            batch_size: Number of traces per batch file
            streaming_mode: Append all traces to a single JSONL file
        """
        self.output_dir = Path(output_dir)
        self.save_individual = save_individual
        self.save_batch = save_batch
        self.batch_size = batch_size
        self.streaming_mode = streaming_mode
        
        self._batch_buffer: List[RolloutTrace] = []
        self._batch_count = 0
        self._trace_count = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Streaming file path
        if streaming_mode:
            self._streaming_file = self.output_dir / "traces.jsonl"
        
        print(f"RolloutTraceSaver initialized: {self.output_dir}")
    
    def save_trace(
        self,
        sample_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        prompt_ids: Optional[List[int]] = None,
        response_ids: Optional[List[int]] = None,
        response_mask: Optional[List[int]] = None,
        prompt_text: Optional[str] = None,
        response_text: Optional[str] = None,
        reward_score: Optional[float] = None,
        turn_scores: Optional[List[float]] = None,
        env_info: Optional[List[Dict[str, Any]]] = None,
        num_user_turns: int = 0,
        num_assistant_turns: int = 0,
        global_step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
        tokenizer=None,
    ) -> RolloutTrace:
        """Save a rollout trace.
        
        Args:
            sample_id: Unique identifier for this sample
            messages: Conversation messages
            prompt_ids: Tokenized prompt IDs
            response_ids: Tokenized response IDs
            response_mask: Response mask (1=LLM, 0=env)
            prompt_text: Decoded prompt text
            response_text: Decoded response text
            reward_score: Final reward score
            turn_scores: Per-turn rewards
            env_info: Environment info from each turn
            num_user_turns: Number of user/env turns
            num_assistant_turns: Number of assistant turns
            global_step: Training step number
            extra: Additional metadata
            tokenizer: Tokenizer for decoding (optional)
            
        Returns:
            The created RolloutTrace object
        """
        # Decode text if tokenizer is provided
        if tokenizer is not None:
            if prompt_text is None and prompt_ids is not None:
                prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            if response_text is None and response_ids is not None:
                response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        trace = RolloutTrace(
            sample_id=sample_id,
            global_step=global_step,
            messages=messages or [],
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            prompt_text=prompt_text,
            response_text=response_text,
            reward_score=reward_score,
            turn_scores=turn_scores or [],
            env_info=env_info or [],
            num_user_turns=num_user_turns,
            num_assistant_turns=num_assistant_turns,
            total_turns=num_user_turns + num_assistant_turns,
            extra=extra or {},
        )
        
        self._trace_count += 1
        
        # Save individual file
        if self.save_individual:
            self._save_individual(trace)
        
        # Streaming mode
        if self.streaming_mode:
            self._append_to_stream(trace)
        
        # Batch mode
        if self.save_batch:
            self._batch_buffer.append(trace)
            if len(self._batch_buffer) >= self.batch_size:
                self._flush_batch()
        
        return trace
    
    def _save_individual(self, trace: RolloutTrace):
        """Save trace as individual JSON file."""
        filename = f"trace_{trace.sample_id}_{self._trace_count}.json"
        filepath = self.output_dir / "individual" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(trace.to_dict(), f, indent=2, default=str)
    
    def _append_to_stream(self, trace: RolloutTrace):
        """Append trace to streaming JSONL file."""
        with open(self._streaming_file, 'a') as f:
            f.write(json.dumps(trace.to_dict(), default=str) + '\n')
    
    def _flush_batch(self):
        """Flush current batch to file."""
        if not self._batch_buffer:
            return
        
        self._batch_count += 1
        filename = f"batch_{self._batch_count:05d}.json"
        filepath = self.output_dir / "batches" / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        batch_data = {
            "batch_id": self._batch_count,
            "num_traces": len(self._batch_buffer),
            "traces": [t.to_dict() for t in self._batch_buffer],
        }
        
        with open(filepath, 'w') as f:
            json.dump(batch_data, f, indent=2, default=str)
        
        self._batch_buffer = []
    
    def flush(self):
        """Flush any pending traces."""
        if self.save_batch and self._batch_buffer:
            self._flush_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get saver statistics."""
        return {
            "total_traces": self._trace_count,
            "total_batches": self._batch_count,
            "pending_in_buffer": len(self._batch_buffer),
            "output_dir": str(self.output_dir),
        }


def init_weave_tracing(
    project_name: str = "opentinker/generic-env",
    experiment_name: str = "default",
    token2text: bool = True,
) -> bool:
    """Initialize Weave (W&B) tracing.
    
    Args:
        project_name: Weave project name (format: entity/project)
        experiment_name: Experiment name for this run
        token2text: Whether to convert token IDs to text in traces
        
    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        from verl.utils.rollout_trace import RolloutTraceConfig
        
        # Check if already initialized
        if RolloutTraceConfig.get_backend() is not None:
            print(f"Tracing already initialized with backend: {RolloutTraceConfig.get_backend()}")
            return True
        
        # Initialize Weave
        RolloutTraceConfig.init(
            project_name=project_name,
            experiment_name=experiment_name,
            backend="weave",
            token2text=token2text,
        )
        
        print(f"✓ Weave tracing initialized: project={project_name}, experiment={experiment_name}")
        return True
        
    except ImportError:
        print("⚠️ Weave not installed. Install with: pip install weave")
        return False
    except Exception as e:
        print(f"⚠️ Failed to initialize Weave tracing: {e}")
        return False


def init_mlflow_tracing(
    project_name: str = "generic-env-training",
    experiment_name: str = "default",
    tracking_uri: Optional[str] = None,
    token2text: bool = True,
) -> bool:
    """Initialize MLflow tracing.
    
    Args:
        project_name: MLflow experiment name
        experiment_name: Run name within the experiment
        tracking_uri: MLflow tracking URI (e.g., sqlite:///traces.db)
        token2text: Whether to convert token IDs to text
        
    Returns:
        True if initialization succeeded, False otherwise
    """
    try:
        import os
        from verl.utils.rollout_trace import RolloutTraceConfig
        
        if RolloutTraceConfig.get_backend() is not None:
            print(f"Tracing already initialized with backend: {RolloutTraceConfig.get_backend()}")
            return True
        
        # Set tracking URI if provided
        if tracking_uri:
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        
        RolloutTraceConfig.init(
            project_name=project_name,
            experiment_name=experiment_name,
            backend="mlflow",
            token2text=token2text,
        )
        
        print(f"✓ MLflow tracing initialized: project={project_name}, experiment={experiment_name}")
        return True
        
    except ImportError:
        print("⚠️ MLflow not installed. Install with: pip install mlflow")
        return False
    except Exception as e:
        print(f"⚠️ Failed to initialize MLflow tracing: {e}")
        return False


# Global instance for easy access
_global_saver: Optional[RolloutTraceSaver] = None


def get_global_saver() -> Optional[RolloutTraceSaver]:
    """Get the global trace saver instance."""
    return _global_saver


def set_global_saver(saver: RolloutTraceSaver):
    """Set the global trace saver instance."""
    global _global_saver
    _global_saver = saver


def init_global_saver(output_dir: str = "/tmp/rollout_traces", **kwargs) -> RolloutTraceSaver:
    """Initialize and set the global trace saver."""
    saver = RolloutTraceSaver(output_dir=output_dir, **kwargs)
    set_global_saver(saver)
    return saver
