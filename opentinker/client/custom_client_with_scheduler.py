#!/usr/bin/env python3
"""
Example: Training with Job Scheduler

This example demonstrates how to use the job scheduler to submit
training jobs without manually managing servers and GPU allocation.
"""

import torch
from transformers import AutoTokenizer
from omegaconf import OmegaConf
import hydra

from http_training_client import ServiceClient, SchedulerClient
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.utils.dataset.rl_dataset import collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from opentinker.environment.environment import RewardFunctionSpec
from opentinker.environment.math.math_env import MathEnvironment
from utils import math_reward_function, resolve_paths_in_config
from scheduler_client_lifecycle import get_lifecycle_manager



@hydra.main(config_path="client_config", config_name="opentinker_param.yaml")
def main(args):
    # Resolve paths to support both absolute and relative paths
    args = resolve_paths_in_config(args)
    
    # Get the lifecycle manager (this automatically enables cleanup handlers)
    lifecycle = get_lifecycle_manager()
    
    print("=" * 60)
    print("Training with Job Scheduler")
    print("=" * 60)
    
    # 1. Connect to scheduler and submit job
    scheduler_url = args.get("scheduler_url", "http://localhost:8765")
    scheduler_api_key = args.get("scheduler_api_key", None)
    
    print(f"\nConnecting to scheduler at {scheduler_url}")
    if scheduler_api_key:
        print("✓ Using API key for authentication")
    else:
        print("⚠ No API key provided - authentication may fail if scheduler requires it")
    
    scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url,
        api_key=scheduler_api_key
    )
    
    # Submit job with configuration
    print("\nSubmitting training job to scheduler...")
    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=args.get("enable_agent_loop", False),
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )
    
    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    
    # Register job for automatic cleanup
    lifecycle.register_job(scheduler_client, job_id)
    
    print(f"\n✓ Job {job_id} allocated!")
    print(f"  Server URL: {server_url}")
    print(f"  GPUs: {job_result.get('gpu_ids')}")
    print(f"  Port: {job_result.get('port')}")
    print("=" * 60)
    
    # 2. Setup environment and reward function
    if args.reward.type == "code":
        reward_function = RewardFunctionSpec(
            type="code",
            code_function=math_reward_function,
        )
    elif args.reward.type == "remote":
        # Auto-start reward server if auto_start is enabled
        if args.reward.remote.get("auto_start", False):
            print("\n" + "="*60)
            print("AUTO-STARTING REWARD SERVER")
            print("="*60)
            
            # Get port from config, or let lifecycle manager auto-assign one
            port = args.reward.remote.get("reward_port", None)
            
            # Start the reward server using lifecycle manager
            _, actual_port = lifecycle.start_reward_server(
                reward_server_path="../reward_functions/math_reward_server.py", 
                reward_ip=args.reward.remote.reward_ip, 
                reward_port=port, 
                wait_time=5
            )
            
            # Build the remote endpoint using the actual port
            remote_endpoint = f"http://{args.reward.remote.reward_ip}:{actual_port}"
            
            print(f"Using reward endpoint: {remote_endpoint}")
            print("="*60 + "\n")
        else:
            # Use the endpoint from config
            port = args.reward.remote.get('reward_port', 30001)
            remote_endpoint = f"http://{args.reward.remote.reward_ip}:{port}"

        # Create RewardFunctionSpec with type="remote"
        reward_function = RewardFunctionSpec(
            type="remote",
            remote_endpoint=remote_endpoint,
            remote_api_key=args.reward.remote.remote_api_key,
        )
        
    env = MathEnvironment(config=args, reward_function=reward_function)
    
    # 3. Connect to allocated server
    print(f"\nConnecting to allocated server at {server_url}")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    
    client.set_config(args, env)
    
    # 4. Train - support both num_steps and num_epochs (num_steps takes precedence)
    num_steps = args.get("num_steps", None)
    num_epochs = args.get("num_epochs", None)
    
    if num_steps:
        print(f"\nStarting training for {num_steps} steps...")
    elif num_epochs:
        print(f"\nStarting training for {num_epochs} epochs...")
    else:
        print("\nStarting training (1 epoch default)...")
        
    print(f"Checkpoint save frequency: {args.save_freq}")
    print(f"Validation frequency: {args.test_freq}")
    print("=" * 60)
    
    
    final_metrics = client.fit(
        env=env,
        num_epochs=num_epochs,
        num_steps=num_steps,
        save_freq=args.save_freq,
        test_freq=args.test_freq,
        verbose=True,
        validate_before_training=True,
    )
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
