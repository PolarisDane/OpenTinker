#!/usr/bin/env python3
"""
Environment API for PPO Training

Provides abstract base class and concrete implementation for configuring
dataloader and reward functions in PPO training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
import inspect

# Note(Siqi):
# ImportError: cannot import name 'ServiceClient'
#  from partially initialized module 'http_training_client'
#   (most likely due to a circular import)


@dataclass
class RewardFunctionSpec:
    """Specification for reward function configuration.

    Supports three types:
    - "config": Load from Python file (path + function name)
    - "remote": Call remote API endpoint (future)
    - "code": Upload custom Python function to server
    """

    type: str  # "config", "remote", or "code"

    # For type="config"
    config_path: Optional[str] = None
    config_name: Optional[str] = None
    config_kwargs: Optional[Dict[str, Any]] = None

    # For type="remote" (future)
    remote_endpoint: Optional[str] = None
    remote_api_key: Optional[str] = None

    # For type="code"
    code_function: Optional[Callable] = None
    code_source: Optional[str] = None

    def __post_init__(self):
        """Validate configuration and extract source code if needed."""
        if self.type not in ["config", "remote", "code"]:
            raise ValueError(
                f"Invalid reward function type: {self.type}. Must be 'config', 'remote', or 'code'"
            )

        if self.type == "config":
            if not self.config_path or not self.config_name:
                raise ValueError(
                    "config_path and config_name are required for type='config'"
                )

        elif self.type == "remote":
            if not self.remote_endpoint:
                raise ValueError("remote_endpoint is required for type='remote'")

        elif self.type == "code":
            if not self.code_function:
                raise ValueError("code_function is required for type='code'")

            # Auto-extract source code if not provided
            if self.code_source is None:
                try:
                    self.code_source = inspect.getsource(self.code_function)
                except (OSError, TypeError) as e:
                    raise ValueError(
                        f"Could not extract source code from function: {e}"
                    )

    def to_config_dict(self) -> Dict[str, Any]:
        """Convert to configuration dictionary for server."""
        if self.type == "config":
            config = {
                "type": "config",
                "config_path": self.config_path,
                "config_name": self.config_name,
            }
            if self.config_kwargs:
                config["config_kwargs"] = self.config_kwargs
            return config

        elif self.type == "remote":
            config = {
                "type": "remote",
                "remote_endpoint": self.remote_endpoint,
            }
            if self.remote_api_key:
                config["remote_api_key"] = self.remote_api_key
            return config

        elif self.type == "code":
            # Use 'name' field to match server config schema (not 'function_name')
            return {
                "type": "code",
                "name": self.code_function.__name__,
            }

        return {}


class BaseEnvironment(ABC):
    """Abstract base class for PPO training environments.

    Subclasses must implement:
    - setup(client): Configure the environment on the server
    - dataloader property: Return the training dataloader
    - get_config(): Return configuration dict for server
    """

    @abstractmethod
    def setup(self, client):
        """Setup environment on the server.

        Args:
            client: HTTPTrainingClient or ServiceClient instance
        """
        pass

    @abstractmethod
    def get_dataloader(self):
        """Return the training dataloader."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for server."""
        pass
