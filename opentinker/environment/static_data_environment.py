# #!/usr/bin/env python3
# """Static Data Environment for LLM Training.

# This module provides an environment wrapper for static datasets (math, reasoning, etc.)
# that integrates with the existing GameEnvironment framework.

# Unlike GameEnvironment which uses AbstractGame + HTTP server for multi-turn interactions,
# StaticDataEnvironment is for single-turn tasks where:
# - Data comes from static files (parquet/jsonl)
# - Reward is computed locally (no HTTP server needed)
# - Each sample is one prompt → one response → done

# Example:
#     from static_data_environment import StaticDataEnvironment
    
#     env = StaticDataEnvironment(
#         config=config,
#         data_paths=['data/math/train.parquet'],
#         reward_function=RewardFunctionSpec(type="code", code_function=my_reward_fn),
#     )
    
#     train_loader, val_loader = env.get_dataloader()
#     client.set_config(config, env)
#     client.fit(env=env, ...)
# """

# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional, Type, Union

# from omegaconf import OmegaConf
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer

# from opentinker.environment.environment import BaseEnvironment, RewardFunctionSpec
# from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
# from opentinker.environment.static_data_generator import StaticDatasetGenerator


# class StaticDataEnvironment(BaseEnvironment):
#     """Environment for single-turn static dataset tasks (math, reasoning, etc.).
    
#     This environment:
#     - Loads data from static files using StaticDatasetGenerator
#     - Uses DynamicGameDataset for tokenization and batching
#     - Supports local reward functions (no HTTP server needed for single-turn)
#     - Provides same interface as GameEnvironment for unified training
    
#     Args:
#         config: Configuration with tokenizer_path, batch_size, etc.
#         data_paths: Path(s) to training data files
#         val_data_paths: Optional path(s) to validation data files
#         reward_function: RewardFunctionSpec for computing rewards
#         prompt_key: Key in data containing prompts (default: "prompt")
#         ground_truth_key: Key for ground truth/answer (default: "answer")
#         data_source: Data source identifier (default: "math")
    
#     Example:
#         config = OmegaConf.create({
#             "tokenizer_path": "Qwen/Qwen2.5-7B-Instruct",
#             "max_prompt_tokens": 1024,
#             "batch_size": 8,
#             "num_steps": 1000,
#         })
        
#         env = StaticDataEnvironment(
#             config=config,
#             data_paths=['data/math/train.parquet'],
#             reward_function=RewardFunctionSpec(type="code", code_function=math_reward),
#         )
#     """
    
#     def __init__(
#         self,
#         config: Dict[str, Any],
#         data_paths: Union[str, List[str]],
#         val_data_paths: Optional[Union[str, List[str]]] = None,
#         reward_function: Optional[RewardFunctionSpec] = None,
#         prompt_key: str = "prompt",
#         ground_truth_key: str = "answer",
#         data_source: str = "math",
#         extra_keys: Optional[List[str]] = None,
#     ):
#         """Initialize StaticDataEnvironment."""
#         self.config = config
#         self.data_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
#         self.val_data_paths = (
#             [val_data_paths] if isinstance(val_data_paths, str) 
#             else list(val_data_paths) if val_data_paths else None
#         )
#         self.reward_function = reward_function
#         self.prompt_key = prompt_key
#         self.ground_truth_key = ground_truth_key
#         self.data_source = data_source
#         self.extra_keys = extra_keys or []
        
#         self.train_dataloader = None
#         self.val_dataloader = None
        
#         self._setup_dataloader()
    
#     def _setup_dataloader(self):
#         """Setup training and validation dataloaders."""
#         # Load tokenizer
#         tokenizer_path = getattr(self.config, 'tokenizer_path', None) or self.config.get('tokenizer_path')
#         print(f"Loading tokenizer from {tokenizer_path}")
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#         tokenizer.padding_side = "left"
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = tokenizer.eos_token
        
#         # Dataset config
#         max_prompt_length = getattr(self.config, 'max_prompt_tokens', 1024)
#         dataset_config = OmegaConf.create({
#             "max_prompt_length": max_prompt_length,
#             "truncation": "right",
#             "return_raw_chat": True,
#         })
        
#         # Calculate virtual size for training
#         batch_size = getattr(self.config, 'batch_size', 8)
#         num_steps = getattr(self.config, 'num_steps', None)
#         num_epochs = getattr(self.config, 'num_epochs', 1)
        
#         # Create training data generator
#         print(f"Creating training data generator from {self.data_paths}")
#         train_generator = StaticDatasetGenerator(
#             data_paths=self.data_paths,
#             interaction_name=self.data_source,
#             prompt_key=self.prompt_key,
#             ground_truth_key=self.ground_truth_key,
#             extra_keys=self.extra_keys,
#             shuffle=True,
#         )
        
#         # Determine virtual size
#         if num_steps:
#             virtual_size = num_steps * batch_size
#         else:
#             virtual_size = len(train_generator) * num_epochs
        
#         print(f"Creating training dataset (virtual_size={virtual_size})")
#         train_dataset = DynamicGameDataset(
#             data_generator=train_generator,
#             tokenizer=tokenizer,
#             config=dataset_config,
#             virtual_size=virtual_size,
#         )
        
#         # Create training dataloader
#         num_workers = getattr(self.config, 'num_workers', 0)
#         print(f"Creating training DataLoader (batch_size={batch_size})")
#         self.train_dataloader = DataLoader(
#             train_dataset,
#             batch_size=batch_size,
#             shuffle=False,  # Shuffling handled by generator
#             num_workers=num_workers,
#             collate_fn=collate_fn,
#             drop_last=True,
#         )
#         print(f"Training dataloader: {len(self.train_dataloader)} batches")
        
#         # Create validation dataloader if paths provided
#         if self.val_data_paths:
#             print(f"Creating validation data generator from {self.val_data_paths}")
#             val_generator = StaticDatasetGenerator(
#                 data_paths=self.val_data_paths,
#                 interaction_name=self.data_source,
#                 prompt_key=self.prompt_key,
#                 ground_truth_key=self.ground_truth_key,
#                 extra_keys=self.extra_keys,
#                 shuffle=False,
#                 seed=42,  # Fixed seed for reproducible validation
#             )
            
#             val_batch_size = getattr(self.config, 'val_batch_size', min(50, len(val_generator)))
#             val_dataset = DynamicGameDataset(
#                 data_generator=val_generator,
#                 tokenizer=tokenizer,
#                 config=dataset_config,
#                 virtual_size=len(val_generator),
#                 seed=42,
#             )
            
#             self.val_dataloader = DataLoader(
#                 val_dataset,
#                 batch_size=val_batch_size,
#                 shuffle=False,
#                 num_workers=num_workers,
#                 collate_fn=collate_fn,
#                 drop_last=False,
#             )
#             print(f"Validation dataloader: {len(self.val_dataloader)} batches")
    
#     def get_dataloader(self):
#         """Return training and validation dataloaders."""
#         return self.train_dataloader, self.val_dataloader
    
#     def get_config(self) -> Dict[str, Any]:
#         """Build configuration dictionary for server."""
#         config = {}
        
#         # Add reward function config
#         if self.reward_function:
#             config["custom_reward_function"] = self.reward_function.to_config_dict()
        
#         return config
    
#     def setup(self, client):
#         """Setup environment on the server.
        
#         For single-turn static data, this mainly handles reward function upload.
#         """
#         # Upload custom reward function code if needed
#         if self.reward_function and self.reward_function.type == "code":
#             print(f"Uploading reward function: {self.reward_function.code_function.__name__}")
#             client.upload_reward_function(
#                 function_name=self.reward_function.code_function.__name__,
#                 source_code=self.reward_function.code_source
#             )
        
#         config = self.get_config()
#         print(f"Environment config: {config}")
#         return config
