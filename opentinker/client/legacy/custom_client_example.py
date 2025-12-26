#!/usr/bin/env python3
import subprocess
import os
import time
import atexit
import requests

from http_training_client import ServiceClient
from opentinker.environment.environment import RewardFunctionSpec
from opentinker.environment.math.math_env_legacy import MathEnvironment
import hydra
from utils import math_reward_function, resolve_paths_in_config, find_free_port


# Global variable to track the reward server process
reward_server_process = None


def start_reward_server(
    reward_server_path="../reward_functions/math_reward_server.py",
    reward_ip="localhost",
    reward_port=None,
    wait_time=5,
):
    """
    Start the math reward server in the background

    Args:
        reward_server_path: Path to the reward server script (relative or absolute)
        reward_ip: IP address for the reward server (default: "localhost")
        reward_port: Port number for the reward server (default: None, auto-assigned)
        wait_time: Time to wait for server to start (default: 5 seconds)

    Returns:
        tuple: (subprocess.Popen, int) - The reward server process and the port number used
    """
    global reward_server_process

    # Auto-assign port if not specified
    if reward_port is None:
        reward_port = find_free_port()

    # Resolve the path to the reward server
    # If it's a relative path, resolve it relative to this script's directory
    if not os.path.isabs(reward_server_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        reward_server_path = os.path.normpath(
            os.path.join(current_dir, reward_server_path)
        )

    if not os.path.exists(reward_server_path):
        raise FileNotFoundError(f"Reward server not found at: {reward_server_path}")

    print(f"Starting reward server from: {reward_server_path}")
    print(f"Server will listen on: http://{reward_ip}:{reward_port}")

    # Start the server process with port argument
    reward_server_process = subprocess.Popen(
        ["python", reward_server_path, "--port", str(reward_port), "--host", reward_ip],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for the server to start
    print(f"Waiting {wait_time} seconds for server to start...")
    time.sleep(wait_time)

    # Check if server is running
    try:
        response = requests.get(f"http://{reward_ip}:{reward_port}/health", timeout=2)
        if response.status_code == 200:
            print("✓ Reward server is running and healthy")
        else:
            print(f"⚠ Reward server responded with status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠ Warning: Could not verify server health: {e}")

    return reward_server_process, reward_port


def cleanup_reward_server():
    """Clean up the reward server process when the client exits"""
    global reward_server_process
    if reward_server_process:
        print("\nShutting down reward server...")
        reward_server_process.terminate()
        try:
            reward_server_process.wait(timeout=5)
            print("✓ Reward server stopped")
        except subprocess.TimeoutExpired:
            print("⚠ Force killing reward server...")
            reward_server_process.kill()


# Register cleanup function to run when script exits
atexit.register(cleanup_reward_server)


@hydra.main(config_path="client_config", config_name="opentinker_param.yaml")
def main(args):
    # Resolve paths to support both absolute and relative paths
    args = resolve_paths_in_config(args)

    if args.reward.type == "code":
        # Create RewardFunctionSpec with type="code"
        reward_function = RewardFunctionSpec(
            type="code",
            code_function=math_reward_function,
        )
    elif args.reward.type == "remote":
        # Auto-start reward server if auto_start is enabled
        if args.reward.remote.get("auto_start", False):
            print("\n" + "=" * 60)
            print("AUTO-STARTING REWARD SERVER")
            print("=" * 60)

            # Get port from config, or let start_reward_server auto-assign one
            port = args.reward.remote.get("reward_port", None)

            # Start the reward server and get the actual port used
            _, actual_port = start_reward_server(
                reward_server_path="../reward_functions/math_reward_server.py",
                reward_ip=args.reward.remote.reward_ip,
                reward_port=port,
                wait_time=5,
            )

            # Build the remote endpoint using the actual port
            remote_endpoint = f"http://{args.reward.remote.reward_ip}:{actual_port}"

            print(f"Using reward endpoint: {remote_endpoint}")
            print("=" * 60 + "\n")
        else:
            # Use the endpoint from config
            port = args.reward.remote.get("reward_port", 30001)
            remote_endpoint = f"http://{args.reward.remote.reward_ip}:{port}"

        # Create RewardFunctionSpec with type="remote"
        reward_function = RewardFunctionSpec(
            type="remote",
            remote_endpoint=remote_endpoint,
            remote_api_key=args.reward.remote.remote_api_key,
        )

    env = MathEnvironment(config=args, reward_function=reward_function)

    print(f"\nConnecting to server at {args.server_url}")
    client = ServiceClient(
        server_url=args.server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )

    client.set_config(args, env)

    # 7. Train - support both num_steps and num_epochs (num_steps takes precedence)
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
