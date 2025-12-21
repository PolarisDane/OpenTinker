#!/usr/bin/env python3
"""Example: Using GenericEnvironment with a Gym-like Environment.

This example demonstrates how to use the GenericEnvironment class
to train an LLM to interact with a Gym-like environment.

The example includes:
1. Setting up the environment configuration
2. Creating interaction specifications
3. Running the training loop

Prerequisites:
1. Start the mock environment server:
   python opentinker/environment/example/mock_env_server.py --port 8080
   
2. Start the training server:
   python opentinker/server/http_training_server.py
"""

from omegaconf import OmegaConf
from opentinker.environment.generic.generic_env import (
    GenericEnvironment,
    InteractionSpec,
    create_gym_environment,
)


def example_basic_usage():
    """Example 1: Basic usage with explicit configuration."""
    
    # Environment configuration
    config = OmegaConf.create({
        "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
        "data_path": "data/train.parquet",
        "val_data_path": None,  # Optional validation data
        "max_prompt_tokens": 1024,
        "max_new_tokens": 512,
        "batch_size": 4,
        "num_workers": 4,
        "algorithm": "agent_loop",  # Required for multi-turn
    })
    
    # Define interaction with the Gym environment
    interaction_specs = [
        InteractionSpec(
            name="text_adventure",
            class_path="verl.interactions.gym_environment_interaction.GymEnvironmentInteraction",
            config={
                "env_endpoint": "http://localhost:8080",
                "max_steps": 50,
                "observation_template": "Environment says: {observation}",
            }
        )
    ]
    
    # Create the environment
    env = GenericEnvironment(config, interaction_specs)
    
    print(f"Environment config: {env.get_config()}")
    print(f"Interaction config path: {env.get_interaction_config_path()}")
    
    # Get dataloaders
    train_dl, val_dl = env.get_dataloader()
    print(f"Training batches: {len(train_dl)}")
    
    return env


def example_convenience_function():
    """Example 2: Using the convenience function."""
    
    config = OmegaConf.create({
        "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
        "data_path": "data/train.parquet",
        "max_prompt_tokens": 1024,
        "max_new_tokens": 512,
        "batch_size": 4,
        "num_workers": 4,
        "algorithm": "agent_loop",
    })
    
    # Quick setup with convenience function
    env = create_gym_environment(
        config=config,
        env_endpoint="http://localhost:8080",
        env_name="my_env",
        max_steps=100,
    )
    
    return env


def example_with_training_client():
    """Example 3: Full integration with training client."""
    
    # This example shows how to integrate with the actual training pipeline
    # Note: Requires http_training_server to be running
    
    from opentinker.client.http_training_client import ServiceClient
    
    # Setup environment
    config = OmegaConf.create({
        "tokenizer_path": "~/models/deepseek-llm-7b-chat",
        "data_path": "data/gsm8k_train.parquet",
        "max_prompt_tokens": 1024,
        "max_new_tokens": 512,
        "batch_size": 8,
        "num_workers": 4,
        "algorithm": "agent_loop",
    })
    
    interaction_specs = [
        InteractionSpec(
            name="math_tutor",
            class_path="verl.interactions.gsm8k_interaction.Gsm8kInteraction",
            config={}
        )
    ]
    
    env = GenericEnvironment(config, interaction_specs)
    
    # Connect to training server
    client = ServiceClient("http://localhost:8000")
    
    try:
        # Setup environment on server
        env_config = env.setup(client)
        
        # Configure server with environment settings
        client.set_generation_config({
            "multi_turn": {
                "interaction_config_path": env.get_interaction_config_path(),
            },
            "agent": {
                "default_agent_loop": "generic_agent",
            },
        })
        
        # Get dataloaders
        train_dl, val_dl = env.get_dataloader()
        
        # Training loop
        for epoch in range(3):
            for batch in train_dl:
                # Submit batch for training
                result = client.submit_batch(batch)
                print(f"Epoch {epoch}: loss={result.get('loss', 'N/A')}")
                
    finally:
        env.cleanup()


def example_dataset_format():
    """Show the expected dataset format for generic environment training.
    
    The dataset should contain:
    - raw_prompt: Initial conversation messages
    - extra_info.interaction_kwargs: Specifies which interaction to use
    
    Example data item:
    {
        "prompt": [
            {"role": "system", "content": "You are playing a text adventure game..."},
            {"role": "user", "content": "You are in a dark cave. Find the treasure."}
        ],
        "extra_info": {
            "interaction_kwargs": {
                "name": "text_adventure"  # Must match an InteractionSpec name
            }
        }
    }
    """
    import json
    
    example_data = {
        "prompt": [
            {"role": "system", "content": "You are an AI playing a text adventure game. Describe your actions clearly."},
            {"role": "user", "content": "You wake up in a dark cave. There's a faint light to the north. What do you do?"}
        ],
        "extra_info": {
            "interaction_kwargs": {
                "name": "text_adventure"
            }
        }
    }
    
    print("Expected dataset format:")
    print(json.dumps(example_data, indent=2))


if __name__ == "__main__":
    print("=" * 60)
    print("GenericEnvironment Examples")
    print("=" * 60)
    
    print("\n--- Example: Dataset Format ---")
    example_dataset_format()
    
    # Uncomment to run with actual data:
    # print("\n--- Example: Basic Usage ---")
    # env = example_basic_usage()
    
    # print("\n--- Example: Convenience Function ---")
    # env = example_convenience_function()
    
    # print("\n--- Example: Full Training ---")
    # example_with_training_client()
