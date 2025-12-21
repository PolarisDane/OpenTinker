
from opentinker.environment.environment import BaseEnvironment, RewardFunctionSpec
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
import inspect
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl.utils.dataset.rl_dataset import collate_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from opentinker.client.utils import prepare_dataset, verify_raw_prompt_format


class MathEnvironment(BaseEnvironment):
    """Concrete environment implementation with config-based initialization.
    
    Example:
        config = {
            "name": "my_environment",
            "dataloader": train_dataloader,
            "reward_function": RewardFunctionSpec(
                type="config",
                config_path="recipe/opentinker/reward_function.py",
                config_name="compute_score_batch"
            ),
        }
        env = MathEnvironment(config)
        env.setup(client)
    """
    
    def __init__(self, config: Dict[str, Any], reward_function: Optional[RewardFunctionSpec] = None):
        """Initialize environment from config dictionary.
        
        Args:
            config: Dictionary with keys:
                - name: Environment name (str)
                - dataloader: PyTorch DataLoader
                - reward_function: RewardFunctionSpec instance
        """
        self.config = config
        self.train_dataloader = None
        self.val_dataloader = None
        self._setup_dataloader()
        self.reward_function = reward_function
        
        if self.reward_function is not None and not isinstance(self.reward_function, RewardFunctionSpec):
            raise ValueError("reward_function must be a RewardFunctionSpec instance")
    
    def _setup_dataloader(self):
        # 1. Load tokenizer
        print(f"Loading tokenizer from {self.config.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer.padding_side = "left"  # Enforce left padding for PPO training

        # 2. Create dataset configuration
        # CRITICAL: return_raw_chat must be True for agent_loop
        data_config = OmegaConf.create({
            "train_files": [self.config.data_path],
            "val_files": [self.config.val_data_path] if self.config.val_data_path else [],
            "prompt_key": "prompt",
            "max_prompt_length": self.config.max_prompt_tokens,  # Shorter for multi-turn
            "max_response_length": self.config.max_new_tokens,  # Per-turn response length
            "truncation": "right",
            "shuffle": True,
            "seed": 42,
            "sampler": None,  # Required by create_rl_sampler
            "return_raw_chat": True,  # REQUIRED for agent_loop
        })

        # 3. Create training dataset
        print(f"Loading training data from {self.config.data_path}")
        train_dataset = prepare_dataset(
            data_paths=[self.config.data_path],
            data_config=data_config,
            tokenizer=tokenizer,
            is_train=True,
        )   
        print(f"Training dataset size: {len(train_dataset)}")

        # 4. Create training dataloader
        print(f"Creating dataloader (batch_size={self.config.batch_size}, num_workers={self.config.num_workers})")
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

        # Verify first batch format
        print("\nVerifying batch format...")
        first_batch = next(iter(self.train_dataloader))
        if self.config.algorithm == "agent_loop":
            verify_raw_prompt_format(first_batch)
        print(f"Sample raw_prompt: {first_batch['raw_prompt'][0][:100]}...")  # Show first 100 chars

        # 5. Create validation dataloader (if provided)
        if self.config.val_data_path:
            print(f"\nLoading validation data from {self.config.val_data_path}")
            val_dataset = prepare_dataset(
                data_paths=[self.config.val_data_path],
                data_config=data_config,
                tokenizer=tokenizer,
                is_train=False,
                max_samples=100,
            )
            print(f"Validation dataset size: {len(val_dataset)}")
            
            self.val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=self.config.val_batch_size if self.config.val_batch_size else len(val_dataset),
                shuffle=True,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn,
                drop_last=False,
            )
            print(f"Validation dataloader created: {len(self.val_dataloader)} batches")

    def get_dataloader(self):
        """Return both training and validation dataloaders."""
        return self.train_dataloader, self.val_dataloader
    
    def get_config(self) -> Dict[str, Any]:
        """Build configuration dictionary for server."""
        config = {}
        
        # Add reward function config if specified
        if self.reward_function:
            reward_config = self.reward_function.to_config_dict()
            # Use top-level custom_reward_function key (not nested in reward_model)
            # This matches the expectation in verl/trainer/ppo/reward.py:get_custom_reward_fn()
            config["custom_reward_function"] = reward_config
        
        return config
    
    def setup(self, client):
        """Setup environment on the server.
        
        For type="code" reward functions, uploads the function code to server.
        For all types, sends configuration to server.
        
        Args:
            client: ServiceClient instance
        """
        # Upload custom reward function code if needed
        if self.reward_function and self.reward_function.type == "code":
            print(f"Uploading custom reward function: {self.reward_function.code_function.__name__}")
            client.upload_reward_function(
                function_name=self.reward_function.code_function.__name__,
                source_code=self.reward_function.code_source
            )

        config = self.get_config()
        return config

if __name__ == "__main__":
    env = MathEnvironment(None)