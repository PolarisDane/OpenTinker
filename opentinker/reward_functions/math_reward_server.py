#!/usr/bin/env python3
"""
Example Remote Reward API Server

This is a simple FastAPI server that implements the reward computation endpoint.
Use this as a template for creating your own remote reward services.

Start the server:
    python remote_reward_api_server.py

The server will listen on http://localhost:30000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from verl.utils.reward_score import default_compute_score
from transformers import PreTrainedTokenizer
from typing import Any


app = FastAPI(
    title="Remote Reward API",
    description="Example reward computation service",
    version="1.0.0"
)


class ComputeRewardRequest(BaseModel):
    """Request model for reward computation"""
    data_source: str
    solution_str: str
    ground_truth: str
    extra_info: Dict[str, Any]
    sandbox_fusion_url: str = None
    concurrent_semaphore: int = None
    memory_limit_mb: int = None
    reward_router_address: str = None
    # reward_model_tokenizer: PreTrainedTokenizer = None
    reward_model_tokenizer: Any = None

class ComputeRewardResponse(BaseModel):
    """Response model for reward computation"""
    reward: float


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "remote_reward_api"}


@app.post("/compute_reward", response_model=ComputeRewardResponse)
async def compute_reward(request: ComputeRewardRequest):
    """
    Compute reward for a single solution.
    
    This is a simple example implementation. Replace with your own logic.
    """
    func_rm_score = default_compute_score(
        request.data_source, 
        request.solution_str, 
        request.ground_truth, 
        request.extra_info,
        # request.sandbox_fusion_url,
        # request.concurrent_semaphore,
        # request.memory_limit_mb,
    )
    
    # Handle both dict and scalar return values
    # default_compute_score may return dict with {"score": ..., other_keys: ...}
    if isinstance(func_rm_score, dict):
        reward = float(func_rm_score.get("score", 0.0))
    else:
        reward = float(func_rm_score)
    
    return ComputeRewardResponse(reward=reward)

    
def main():
    """Start the remote reward API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Remote Reward API Server")
    parser.add_argument("--port", type=int, default=30001, help="Port to listen on (default: 30001)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args()
    
    print("="*60)
    print("Starting Remote Reward API Server")
    print("="*60)
    print("Endpoints:")
    print(f"  - Health: http://localhost:{args.port}/health")
    print(f"  - Compute reward: http://localhost:{args.port}/compute_reward")
    print("="*60)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
