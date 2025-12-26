#!/usr/bin/env python3
"""
Example: Training with Generic Environment (LLM-Environment Interaction)

This example demonstrates how to use GenericEnvironment with the job scheduler
for training LLMs to interact with external environments (like OpenAI Gym).

Key differences from MathEnvironment:
- No external reward function needed (reward comes from environment)
- Uses GenericAgentLoop instead of ToolAgentLoop
- Configures interaction via InteractionSpec

Usage:
    1. Start the mock environment server:
       python opentinker/environment/example/mock_env_server.py --port 8080

    2. Start the scheduler:
       python opentinker/scheduler/launch_scheduler.py

    3. Run this client:
       python generic_env_client.py
"""

from omegaconf import OmegaConf
import hydra

from http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.generic.generic_env import (
    GenericEnvironment,
    InteractionSpec,
)
from utils import resolve_paths_in_config
from scheduler_client_lifecycle import get_lifecycle_manager


@hydra.main(config_path="client_config", config_name="generic_env_param.yaml")
def main(args):
    # Resolve paths to support both absolute and relative paths
    args = resolve_paths_in_config(args)

    # Get the lifecycle manager (this automatically enables cleanup handlers)
    lifecycle = get_lifecycle_manager()

    # Initialize Weave tracing (optional, requires wandb/weave installed)
    enable_tracing = args.get("enable_tracing", False)
    if enable_tracing:
        try:
            from opentinker.utils.rollout_trace_saver import init_weave_tracing

            weave_project = args.get("weave_project", "generic-env-test")
            init_weave_tracing(
                project_name=weave_project,
                experiment_name=args.experiment_name,
                token2text=True,
            )
        except Exception as e:
            print(f"⚠ Failed to initialize Weave tracing: {e}")

    print("=" * 60)
    print("Training with Generic Environment (LLM-Environment Interaction)")
    print("=" * 60)

    # 1. Connect to scheduler and submit job
    scheduler_url = args.get("scheduler_url", "http://localhost:8765")
    scheduler_api_key = args.get("scheduler_api_key", None)

    print(f"\nConnecting to scheduler at {scheduler_url}")
    if scheduler_api_key:
        print("✓ Using API key for authentication")
    else:
        print(
            "⚠ No API key provided - authentication may fail if scheduler requires it"
        )

    scheduler_client = SchedulerClient(
        scheduler_url=scheduler_url, api_key=scheduler_api_key
    )

    # Submit job with configuration
    # IMPORTANT: enable_agent_loop must be True for GenericEnvironment
    print("\nSubmitting training job to scheduler...")
    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=True,  # REQUIRED for GenericAgentLoop
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

    # 2. Setup GenericEnvironment with interaction specs
    # Unlike MathEnvironment, GenericEnvironment doesn't need a reward_function
    # because the environment provides rewards via the interaction

    interaction_specs = [
        InteractionSpec(
            name=args.interaction.name,
            class_path=args.interaction.class_path,
            config=OmegaConf.to_container(args.interaction.config, resolve=True),
        )
    ]

    print("\nSetting up GenericEnvironment...")
    print(f"  Interaction: {args.interaction.name}")
    print(f"  Class: {args.interaction.class_path}")
    print(f"  Config: {dict(args.interaction.config)}")

    env = GenericEnvironment(
        config=args,
        interaction_specs=interaction_specs,
        reward_function=None,  # No external reward function needed!
    )

    print("✓ Environment created")
    print(f"  Interaction config path: {env.get_interaction_config_path()}")

    # 3. Connect to allocated server
    print(f"\nConnecting to allocated server at {server_url}")
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

    # Set configuration on server
    # This includes the interaction_config_path for GenericAgentLoop
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

    try:
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

    finally:
        # Clean up temporary files
        env.cleanup()


if __name__ == "__main__":
    main()
