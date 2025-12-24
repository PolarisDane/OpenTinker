#!/usr/bin/env python3
"""Vision-Language Static Data Generator for OpenTinker.

This module extends StaticDatasetGenerator to support vision-language models
by loading and processing images from parquet files.
"""

import logging
from typing import Any, Dict, List, Optional

from opentinker.environment.static_data_generator import StaticDatasetGenerator

logger = logging.getLogger(__name__)


class StaticDatasetGeneratorVL(StaticDatasetGenerator):
    """Static dataset generator with vision-language support.
    
    This generator extends StaticDatasetGenerator to handle image data
    from parquet files. Images are typically stored as lists of PIL images
    or image paths in the dataset.
    
    Args:
        data_paths: List of parquet file paths
        interaction_name: Name of the interaction handler
        prompt_key: Key for prompt field in data (default: "prompt")
        ground_truth_key: Key for ground truth answer (default: "ground_truth")
        image_key: Key for image field in data (default: "images")
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling
        system_prompt: Optional system prompt to prepend
    
    Example:
        generator = StaticDatasetGeneratorVL(
            data_paths=["~/data/geo3k/train.parquet"],
            interaction_name="game",
            image_key="images",
        )
    """
    
    def __init__(
        self,
        data_paths: List[str],
        interaction_name: str = "game",
        prompt_key: str = "prompt",
        ground_truth_key: str = "ground_truth",
        image_key: str = "images",
        shuffle: bool = False,
        seed: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ):
        super().__init__(
            data_paths=data_paths,
            interaction_name=interaction_name,
            prompt_key=prompt_key,
            ground_truth_key=ground_truth_key,
            shuffle=shuffle,
            seed=seed,
            system_prompt=system_prompt,
        )
        self.image_key = image_key
        logger.info(f"StaticDatasetGeneratorVL initialized with image_key='{image_key}'")
    
    def generate_sample(self, index: int) -> Dict[str, Any]:
        """Generate a sample with vision-language data.
        
        Args:
            index: Sample index
        
        Returns:
            Dict with keys:
                - prompt: List of message dicts
                - env_kwargs: Dict with ground_truth
                - images: List of images (if present)
                - data_source: Data source identifier
        """
        # Get base sample from parent class
        sample = super().generate_sample(index)
        
        # Add images if present in the data
        actual_idx = self._indices[index % len(self._samples)]
        row = self._samples[actual_idx]
        if self.image_key in row:
            images = row[self.image_key]
            # Ensure images is a list
            if not isinstance(images, list):
                images = [images] if images is not None else []
            sample["images"] = images
        else:
            sample["images"] = []
        
        return sample
