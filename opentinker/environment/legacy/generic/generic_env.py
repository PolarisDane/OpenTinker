#!/usr/bin/env python3
"""Generic Environment for LLM-Environment Interaction Training.

This module provides a generic environment implementation following the BaseEnvironment
pattern. It's designed to work with GenericAgentLoop for multi-turn LLM-environment
interaction training where the environment provides its own rewards.

Key Features:
- No external reward function needed (reward comes from environment)
- Supports interaction_config for BaseInteraction subclasses
- Works with any Gym-like environment via GymEnvironmentInteraction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import os

from transformers import AutoTokenizer
from omegaconf import OmegaConf

from opentinker.environment.environment import BaseEnvironment, RewardFunctionSpec
from verl.utils.dataset.rl_dataset import collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.main_ppo import create_rl_sampler
from opentinker.client.utils.utils import prepare_dataset, verify_raw_prompt_format


@dataclass
class InteractionSpec:
    """Specification for interaction configuration.

    This defines how the LLM interacts with the environment during training.
    The interaction handles environment step logic (reset, step, reward).

    Attributes:
        name: Unique name for this interaction (used in interaction_kwargs)
        class_path: Full path to the interaction class
        config: Configuration dict passed to the interaction class
    """

    name: str
    class_path: str  # e.g., "verl.interactions.gym_environment_interaction.GymEnvironmentInteraction"
    config: Dict[str, Any] = field(default_factory=dict)

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to YAML-compatible configuration dict.

        Format expected by interaction_registry.py:
        - class_name: Full path to interaction class
        - config: Configuration dict
        - name: Interaction name (optional, but we always provide it)
        """
        return {
            "name": self.name,
            "class_name": self.class_path,  # Note: 'class_name' not 'class'
            "config": self.config,
        }


class GenericEnvironment(BaseEnvironment):
    """Generic environment for multi-turn LLM-environment interaction.

    This environment is designed for training LLMs to interact with external
    environments (like OpenAI Gym). The environment provides rewards through
    the interaction, so no external reward function is needed.

    Configuration:
        config.tokenizer_path: Path to tokenizer
        config.data_path: Path to training data
        config.val_data_path: Optional path to validation data
        config.max_prompt_tokens: Maximum prompt length
        config.max_new_tokens: Maximum response length per turn
        config.batch_size: Training batch size
        config.num_workers: DataLoader workers
        config.algorithm: Should be "agent_loop" for multi-turn

    Interaction Configuration:
        The environment uses BaseInteraction subclasses to handle step logic.
        Configure via interaction_specs parameter.

    Example:
        config = OmegaConf.create({
            "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
            "data_path": "data/train.parquet",
            "max_prompt_tokens": 1024,
            "max_new_tokens": 512,
            "batch_size": 4,
            "num_workers": 4,
            "algorithm": "agent_loop",
        })

        interaction_specs = [
            InteractionSpec(
                name="gym_env",
                class_path="verl.interactions.gym_environment_interaction.GymEnvironmentInteraction",
                config={"env_endpoint": "http://localhost:8080", "max_steps": 100}
            )
        ]

        env = GenericEnvironment(config, interaction_specs)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        interaction_specs: Optional[List[InteractionSpec]] = None,
        reward_function: Optional[RewardFunctionSpec] = None,
    ):
        """Initialize the generic environment.

        Args:
            config: Environment configuration (OmegaConf or dict)
            interaction_specs: List of interaction specifications
            reward_function: Optional external reward function (usually None for
                           environments that provide their own rewards)
        """
        self.config = config
        self.interaction_specs = interaction_specs or []
        self.reward_function = reward_function

        self.train_dataloader = None
        self.val_dataloader = None
        self._interaction_config_path = None

        self._setup_dataloader()
        self._setup_interaction_config()

        if self.reward_function is not None and not isinstance(
            self.reward_function, RewardFunctionSpec
        ):
            raise ValueError("reward_function must be a RewardFunctionSpec instance")

    def _setup_dataloader(self):
        """Setup training and validation dataloaders."""
        # Load tokenizer
        print(f"Loading tokenizer from {self.config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer.padding_side = "left"  # Required for PPO training

        # Create dataset configuration
        # CRITICAL: return_raw_chat must be True for agent_loop
        data_config = OmegaConf.create(
            {
                "train_files": [self.config.data_path],
                "val_files": [self.config.val_data_path]
                if getattr(self.config, "val_data_path", None)
                else [],
                "prompt_key": getattr(self.config, "prompt_key", "prompt"),
                "max_prompt_length": self.config.max_prompt_tokens,
                "max_response_length": self.config.max_new_tokens,
                "truncation": "right",
                "shuffle": True,
                "seed": 42,
                "sampler": None,
                "return_raw_chat": True,  # REQUIRED for agent_loop
            }
        )

        # Create training dataset
        print(f"Loading training data from {self.config.data_path}")
        train_dataset = prepare_dataset(
            data_paths=[self.config.data_path],
            data_config=data_config,
            tokenizer=tokenizer,
            is_train=True,
        )
        print(f"Training dataset size: {len(train_dataset)}")

        # Create training dataloader
        print(
            f"Creating dataloader (batch_size={self.config.batch_size}, num_workers={self.config.num_workers})"
        )
        self.train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            sampler=create_rl_sampler(data_config, train_dataset),
        )
        print(f"Dataloader created: {len(self.train_dataloader)} batches")

        # Verify batch format for agent_loop
        print("\nVerifying batch format...")
        first_batch = next(iter(self.train_dataloader))
        if getattr(self.config, "algorithm", None) == "agent_loop":
            verify_raw_prompt_format(first_batch)
        print(
            f"Sample raw_prompt: {str(first_batch.get('raw_prompt', ['N/A'])[0])[:100]}..."
        )

        # Create validation dataloader if provided
        val_data_path = getattr(self.config, "val_data_path", None)
        if val_data_path:
            print(f"\nLoading validation data from {val_data_path}")

            # Smart default: use val_batch_size as max_samples if val_max_samples not specified
            # This allows users to only set val_batch_size to control both
            val_batch_size = getattr(self.config, "val_batch_size", None)
            val_max_samples = getattr(self.config, "val_max_samples", None)

            # Priority: val_max_samples > val_batch_size > 100 (legacy default)
            if val_max_samples is None:
                val_max_samples = val_batch_size if val_batch_size else 100

            val_dataset = prepare_dataset(
                data_paths=[val_data_path],
                data_config=data_config,
                tokenizer=tokenizer,
                is_train=False,
                max_samples=val_max_samples,
            )
            print(f"Validation dataset size: {len(val_dataset)}")

            # Use val_batch_size if set, otherwise use full dataset
            val_batch_size = val_batch_size or len(val_dataset)
            self.val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=val_batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn,
                drop_last=False,
            )
            print(f"Validation dataloader created: {len(self.val_dataloader)} batches")

    def _setup_interaction_config(self):
        """Generate interaction config YAML file from interaction_specs.

        Also stores the content for cross-node transmission.

        The format expected by interaction_registry.py is:
        interaction:
          - name: interaction_name
            class_name: full.path.to.InteractionClass
            config:
              key: value
        """
        if not self.interaction_specs:
            return

        import tempfile
        import yaml

        # Convert specs to list format
        interaction_list = [spec.to_config_dict() for spec in self.interaction_specs]

        # Wrap in 'interaction' key as expected by initialize_interactions_from_config
        config_dict = {"interaction": interaction_list}

        # Store the config content as YAML string for cross-node transmission
        self._interaction_config_content = yaml.dump(
            config_dict, default_flow_style=False
        )

        # Write to temporary file (for backward compatibility)
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="interaction_config_")
        with os.fdopen(fd, "w") as f:
            f.write(self._interaction_config_content)

        self._interaction_config_path = path
        print(f"Generated interaction config at: {path}")

    def get_dataloader(self):
        """Return both training and validation dataloaders."""
        return self.train_dataloader, self.val_dataloader

    def get_interaction_config_path(self) -> Optional[str]:
        """Return path to the generated interaction config file."""
        return self._interaction_config_path

    def get_interaction_config_content(self) -> Optional[str]:
        """Return interaction config content as YAML string for cross-node transmission."""
        return getattr(self, "_interaction_config_content", None)

    def get_config(self) -> Dict[str, Any]:
        """Build configuration dictionary for server.

        Returns config for agent_loop with interaction support.
        Includes both path (for local use) and content (for cross-node distribution).
        """
        config = {}

        # Add agent loop configuration in the nested structure expected by server
        if self._interaction_config_path:
            config["actor_rollout_ref"] = {
                "rollout": {
                    "multi_turn": {
                        "interaction_config_path": self._interaction_config_path,
                        # Include content for cross-node transmission
                        "interaction_config_content": getattr(
                            self, "_interaction_config_content", None
                        ),
                    },
                    "agent": {
                        "default_agent_loop": "generic_agent",
                    },
                },
            }

        # Add reward function config if explicitly specified
        # (usually not needed as environment provides rewards)
        if self.reward_function:
            config["custom_reward_function"] = self.reward_function.to_config_dict()

        return config

    def setup(self, client):
        """Setup environment on the server.

        For generic environments, this primarily sets up the interaction
        configuration. No reward function upload is typically needed.

        Args:
            client: ServiceClient instance
        """
        # Upload custom reward function if explicitly provided
        if self.reward_function and self.reward_function.type == "code":
            print(
                f"Uploading custom reward function: {self.reward_function.code_function.__name__}"
            )
            client.upload_reward_function(
                function_name=self.reward_function.code_function.__name__,
                source_code=self.reward_function.code_source,
            )

        config = self.get_config()
        print(f"Environment config: {config}")
        return config

    def cleanup(self):
        """Clean up temporary files."""
        if self._interaction_config_path and os.path.exists(
            self._interaction_config_path
        ):
            os.remove(self._interaction_config_path)
            print(
                f"Removed temporary interaction config: {self._interaction_config_path}"
            )


# Convenience function for creating simple environments
def create_gym_environment(
    config: Dict[str, Any],
    env_endpoint: str,
    env_name: str = "gym_env",
    max_steps: int = 100,
    observation_template: str = "{observation}",
) -> GenericEnvironment:
    """Convenience function to create a GenericEnvironment with Gym interaction.

    Args:
        config: Base environment configuration
        env_endpoint: HTTP endpoint for the Gym environment server
        env_name: Name for the interaction (used in interaction_kwargs)
        max_steps: Maximum steps per episode
        observation_template: Template for formatting observations

    Returns:
        Configured GenericEnvironment instance
    """
    interaction_specs = [
        InteractionSpec(
            name=env_name,
            class_path="verl.interactions.gym_environment_interaction.GymEnvironmentInteraction",
            config={
                "env_endpoint": env_endpoint,
                "max_steps": max_steps,
                "observation_template": observation_template,
            },
        )
    ]

    return GenericEnvironment(config, interaction_specs)
