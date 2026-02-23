"""Task set configuration manager for Android World.

Loads task_sets.yaml and provides split-aware task sampling for
training and evaluation (ID / OOD).

Usage:
    config = TaskSetConfig()                        # loads default yaml
    config = TaskSetConfig("path/to/task_sets.yaml")

    train_tasks = config.get_tasks("train")         # list[str]
    test_id     = config.get_tasks("test_id")
    test_ood    = config.get_tasks("test_ood")

    task = config.sample_task("train")              # random single task
    tasks = config.sample_tasks("test_ood", k=5)    # random k tasks
"""

from __future__ import annotations

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config lives next to this file
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "task_sets.yaml")

# Valid split names
VALID_SPLITS = ("train", "test_id", "test_ood")


class TaskSetConfig:
    """Manages train / test_id / test_ood task splits."""

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path or _DEFAULT_CONFIG_PATH
        self._raw: Dict[str, Any] = {}
        self._task_lists: Dict[str, List[str]] = {}
        self._eval_settings: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self):
        path = Path(self._config_path)
        if not path.exists():
            raise FileNotFoundError(f"Task set config not found: {path}")

        with open(path, "r") as f:
            self._raw = yaml.safe_load(f) or {}

        # Train tasks
        train_tasks = self._raw.get("train", {}).get("tasks", [])
        if not train_tasks:
            logger.warning("No training tasks defined in config – using fallback.")
            train_tasks = ["ContactsAddContact"]
        self._task_lists["train"] = list(train_tasks)

        # Test ID
        test_id_cfg = self._raw.get("test_id", {})
        if test_id_cfg.get("use_train_tasks", False):
            id_tasks = test_id_cfg.get("tasks", [])
            if id_tasks:
                # subset of train tasks
                self._task_lists["test_id"] = list(id_tasks)
            else:
                self._task_lists["test_id"] = list(train_tasks)
        else:
            self._task_lists["test_id"] = list(
                test_id_cfg.get("tasks", train_tasks)
            )

        # Test OOD
        ood_tasks = self._raw.get("test_ood", {}).get("tasks", [])
        if not ood_tasks:
            logger.warning("No OOD test tasks defined.")
        self._task_lists["test_ood"] = list(ood_tasks)

        # Eval settings
        self._eval_settings = self._raw.get("eval_settings", {})

        # Validate: train and OOD should be disjoint
        overlap = set(self._task_lists["train"]) & set(self._task_lists["test_ood"])
        if overlap:
            logger.warning(
                f"Train/OOD overlap detected ({len(overlap)} tasks): {overlap}. "
                "This defeats the purpose of OOD testing!"
            )

        logger.info(
            f"TaskSetConfig loaded: train={len(self._task_lists['train'])}, "
            f"test_id={len(self._task_lists['test_id'])}, "
            f"test_ood={len(self._task_lists['test_ood'])}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tasks(self, split: str) -> List[str]:
        """Return the full task list for a split."""
        self._check_split(split)
        return list(self._task_lists[split])

    def sample_task(self, split: str, rng: Optional[random.Random] = None) -> str:
        """Sample a single random task from a split."""
        self._check_split(split)
        tasks = self._task_lists[split]
        if not tasks:
            raise ValueError(f"No tasks available in split '{split}'")
        r = rng or random
        return r.choice(tasks)

    def sample_tasks(
        self,
        split: str,
        k: int = 1,
        replace: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        """Sample k tasks from a split (with or without replacement)."""
        self._check_split(split)
        tasks = self._task_lists[split]
        if not tasks:
            raise ValueError(f"No tasks available in split '{split}'")
        r = rng or random
        if replace:
            return [r.choice(tasks) for _ in range(k)]
        else:
            return r.sample(tasks, min(k, len(tasks)))

    @property
    def eval_settings(self) -> Dict[str, Any]:
        """Eval settings from config (n_instances_per_task, max_steps, seed)."""
        return dict(self._eval_settings)

    @property
    def n_instances_per_task(self) -> int:
        return self._eval_settings.get("n_instances_per_task", 3)

    @property
    def eval_max_steps(self) -> int:
        return self._eval_settings.get("max_steps", 30)

    @property
    def eval_seed(self) -> Optional[int]:
        return self._eval_settings.get("seed", None)

    @property
    def config_path(self) -> str:
        return self._config_path

    def num_tasks(self, split: str) -> int:
        self._check_split(split)
        return len(self._task_lists[split])

    def validate_against_registry(self, available_tasks: Dict[str, Any]) -> Dict[str, List[str]]:
        """Validate all configured tasks exist in the AndroidWorld registry.

        Args:
            available_tasks: dict from TaskRegistry.get_registry()

        Returns:
            Dict with 'valid' and 'invalid' task names per split.
        """
        report: Dict[str, List[str]] = {}
        for split in VALID_SPLITS:
            invalid = [t for t in self._task_lists[split] if t not in available_tasks]
            if invalid:
                logger.warning(f"[{split}] {len(invalid)} tasks not in registry: {invalid}")
            report[split] = invalid
        return report

    def summary(self) -> str:
        """Human-readable summary."""
        lines = ["=== Android World Task Set Config ==="]
        for split in VALID_SPLITS:
            tasks = self._task_lists[split]
            lines.append(f"  {split:12s}: {len(tasks)} tasks")
        overlap = set(self._task_lists["train"]) & set(self._task_lists["test_ood"])
        lines.append(f"  Train/OOD overlap: {len(overlap)} (should be 0)")
        lines.append(f"  Eval instances/task: {self.n_instances_per_task}")
        lines.append(f"  Eval max steps: {self.eval_max_steps}")
        lines.append(f"  Eval seed: {self.eval_seed}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_split(self, split: str):
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. Must be one of {VALID_SPLITS}"
            )

    def __repr__(self):
        return (
            f"TaskSetConfig(train={len(self._task_lists.get('train', []))}, "
            f"test_id={len(self._task_lists.get('test_id', []))}, "
            f"test_ood={len(self._task_lists.get('test_ood', []))})"
        )
