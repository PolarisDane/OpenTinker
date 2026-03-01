"""Evaluation framework for Android World.

Provides comprehensive evaluation metrics including:
  - Per-task and aggregate success rate
  - Average / median steps to completion
  - Average reward
  - Per-app-category breakdown
  - Timeout rate
  - Invalid action rate

Supports two evaluation modes:
  1. **Live evaluation**: Runs episodes on a real/mock AndroidWorld env.
  2. **Log-based evaluation**: Computes metrics from saved episode logs.

Usage (live):
    evaluator = AndroidWorldEvaluator(
        game=game,
        task_set_config=config,
        split="test_ood",
    )
    results = evaluator.run()
    print(results.summary())

Usage (from logs):
    results = EvalResults.from_episode_logs(logs)
    print(results.summary())
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from opentinker.environment.android_world.android_world_game import AndroidWorldGame
    from opentinker.environment.android_world.task_set_config import TaskSetConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""

    task_name: str
    instance_id: int  # which instance of this task type (0, 1, 2, ...)
    success: bool
    total_steps: int
    total_reward: float
    timeout: bool  # whether episode ended by hitting max_steps
    invalid_actions: int  # number of invalid action penalties
    wall_time_seconds: float  # wall-clock time for episode
    seed: Optional[int] = None
    error: Optional[str] = None  # error message if episode failed abnormally

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskMetrics:
    """Aggregated metrics for a single task type."""

    task_name: str
    n_episodes: int = 0
    n_success: int = 0
    n_timeout: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    total_invalid_actions: int = 0
    total_wall_time: float = 0.0
    steps_on_success: List[int] = field(default_factory=list)
    n_errors: int = 0

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_episodes if self.n_episodes > 0 else 0.0

    @property
    def timeout_rate(self) -> float:
        return self.n_timeout / self.n_episodes if self.n_episodes > 0 else 0.0

    @property
    def avg_steps(self) -> float:
        return self.total_steps / self.n_episodes if self.n_episodes > 0 else 0.0

    @property
    def avg_steps_on_success(self) -> float:
        return (
            sum(self.steps_on_success) / len(self.steps_on_success)
            if self.steps_on_success
            else 0.0
        )

    @property
    def median_steps_on_success(self) -> float:
        if not self.steps_on_success:
            return 0.0
        s = sorted(self.steps_on_success)
        n = len(s)
        if n % 2 == 1:
            return float(s[n // 2])
        return (s[n // 2 - 1] + s[n // 2]) / 2.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / self.n_episodes if self.n_episodes > 0 else 0.0

    @property
    def avg_invalid_actions(self) -> float:
        return (
            self.total_invalid_actions / self.n_episodes if self.n_episodes > 0 else 0.0
        )

    @property
    def avg_wall_time(self) -> float:
        return self.total_wall_time / self.n_episodes if self.n_episodes > 0 else 0.0

    def add_episode(self, ep: EpisodeResult):
        self.n_episodes += 1
        self.n_success += int(ep.success)
        self.n_timeout += int(ep.timeout)
        self.total_steps += ep.total_steps
        self.total_reward += ep.total_reward
        self.total_invalid_actions += ep.invalid_actions
        self.total_wall_time += ep.wall_time_seconds
        if ep.success:
            self.steps_on_success.append(ep.total_steps)
        if ep.error:
            self.n_errors += 1


# ============================================================================
# App category mapping (for per-category breakdown)
# ============================================================================

_APP_CATEGORY = {
    # Contacts
    "ContactsAddContact": "Contacts",
    "ContactsNewContactDraft": "Contacts",
    # Calendar
    "SimpleCalendarAddOneEvent": "Calendar",
    "SimpleCalendarAddOneEventInTwoWeeks": "Calendar",
    "SimpleCalendarAddOneEventRelativeDay": "Calendar",
    "SimpleCalendarAddOneEventTomorrow": "Calendar",
    "SimpleCalendarAddRepeatingEvent": "Calendar",
    "SimpleCalendarDeleteEvents": "Calendar",
    "SimpleCalendarDeleteEventsOnRelativeDay": "Calendar",
    "SimpleCalendarDeleteOneEvent": "Calendar",
    # SMS
    "SimpleSmsSend": "SMS",
    "SimpleSmsReply": "SMS",
    "SimpleSmsReplyMostRecent": "SMS",
    "SimpleSmsResend": "SMS",
    "SimpleSmsSendClipboardContent": "SMS",
    "SimpleSmsSendReceivedAddress": "SMS",
    # Markor
    "MarkorCreateNote": "Markor",
    "MarkorDeleteNote": "Markor",
    "MarkorEditNote": "Markor",
    "MarkorCreateFolder": "Markor",
    "MarkorAddNoteHeader": "Markor",
    "MarkorChangeNoteContent": "Markor",
    "MarkorCreateNoteFromClipboard": "Markor",
    "MarkorDeleteAllNotes": "Markor",
    "MarkorDeleteNewestNote": "Markor",
    "MarkorMergeNotes": "Markor",
    "MarkorMoveNote": "Markor",
    "MarkorTranscribeReceipt": "Markor",
    "MarkorTranscribeVideo": "Markor",
    # Composite
    "MarkorCreateNoteAndSms": "Composite",
    "TurnOffWifiAndTurnOnBluetooth": "Composite",
    "TurnOnWifiAndOpenApp": "Composite",
    # System
    "OpenAppTaskEval": "System",
    "SystemWifiTurnOn": "System",
    "SystemWifiTurnOff": "System",
    "SystemWifiTurnOnVerify": "System",
    "SystemWifiTurnOffVerify": "System",
    "SystemBluetoothTurnOn": "System",
    "SystemBluetoothTurnOff": "System",
    "SystemBluetoothTurnOnVerify": "System",
    "SystemBluetoothTurnOffVerify": "System",
    "SystemBrightnessMax": "System",
    "SystemBrightnessMin": "System",
    "SystemBrightnessMaxVerify": "System",
    "SystemBrightnessMinVerify": "System",
    "SystemCopyToClipboard": "System",
    # Expense
    "ExpenseAddSingle": "Expense",
    "ExpenseDeleteSingle": "Expense",
    "ExpenseAddMultiple": "Expense",
    "ExpenseAddMultipleFromGallery": "Expense",
    "ExpenseAddMultipleFromMarkor": "Expense",
    "ExpenseDeleteDuplicates": "Expense",
    "ExpenseDeleteDuplicates2": "Expense",
    "ExpenseDeleteMultiple": "Expense",
    "ExpenseDeleteMultiple2": "Expense",
    # Files
    "FilesDeleteFile": "Files",
    "FilesMoveFile": "Files",
    # Recipe
    "RecipeAddSingleRecipe": "Recipe",
    "RecipeAddMultipleRecipes": "Recipe",
    "RecipeAddMultipleRecipesFromImage": "Recipe",
    "RecipeAddMultipleRecipesFromMarkor": "Recipe",
    "RecipeAddMultipleRecipesFromMarkor2": "Recipe",
    "RecipeDeleteSingleRecipe": "Recipe",
    "RecipeDeleteMultipleRecipes": "Recipe",
    "RecipeDeleteMultipleRecipesWithConstraint": "Recipe",
    "RecipeDeleteMultipleRecipesWithNoise": "Recipe",
    "RecipeDeleteDuplicateRecipes": "Recipe",
    "RecipeDeleteDuplicateRecipes2": "Recipe",
    "RecipeDeleteDuplicateRecipes3": "Recipe",
    "RecipeDeleteSingleWithRecipeWithNoise": "Recipe",
    # Browser
    "BrowserDraw": "Browser",
    "BrowserMaze": "Browser",
    "BrowserMultiply": "Browser",
    # Audio
    "AudioRecorderRecordAudio": "AudioRecorder",
    "AudioRecorderRecordAudioWithFileName": "AudioRecorder",
    # Camera
    "CameraTakePhoto": "Camera",
    "CameraTakeVideo": "Camera",
    # Clock
    "ClockStopWatchPausedVerify": "Clock",
    "ClockStopWatchRunning": "Clock",
    "ClockTimerEntry": "Clock",
    # OsmAnd
    "OsmAndFavorite": "OsmAnd",
    "OsmAndMarker": "OsmAnd",
    "OsmAndTrack": "OsmAnd",
    # Retro Music
    "RetroCreatePlaylist": "RetroMusic",
    "RetroPlayingQueue": "RetroMusic",
    "RetroPlaylistDuration": "RetroMusic",
    "RetroSavePlaylist": "RetroMusic",
    # Drawing
    "SimpleDrawProCreateDrawing": "SimpleDraw",
    # Gallery
    "SaveCopyOfReceiptTaskEval": "Gallery",
    # VLC
    "VlcCreatePlaylist": "VLC",
    "VlcCreateTwoPlaylists": "VLC",
}


def get_app_category(task_name: str) -> str:
    """Get the app category for a task name."""
    return _APP_CATEGORY.get(task_name, "Unknown")


# ============================================================================
# EvalResults — aggregate container
# ============================================================================


class EvalResults:
    """Aggregated evaluation results with rich reporting."""

    def __init__(self, split: str):
        self.split = split
        self.episodes: List[EpisodeResult] = []
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def add_episode(self, ep: EpisodeResult):
        self.episodes.append(ep)
        if ep.task_name not in self.task_metrics:
            self.task_metrics[ep.task_name] = TaskMetrics(task_name=ep.task_name)
        self.task_metrics[ep.task_name].add_episode(ep)

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def n_tasks(self) -> int:
        return len(self.task_metrics)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.success for e in self.episodes) / len(self.episodes)

    @property
    def avg_steps(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.total_steps for e in self.episodes) / len(self.episodes)

    @property
    def avg_steps_on_success(self) -> float:
        succ = [e.total_steps for e in self.episodes if e.success]
        return sum(succ) / len(succ) if succ else 0.0

    @property
    def median_steps_on_success(self) -> float:
        succ = sorted(e.total_steps for e in self.episodes if e.success)
        if not succ:
            return 0.0
        n = len(succ)
        if n % 2 == 1:
            return float(succ[n // 2])
        return (succ[n // 2 - 1] + succ[n // 2]) / 2.0

    @property
    def avg_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.total_reward for e in self.episodes) / len(self.episodes)

    @property
    def timeout_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.timeout for e in self.episodes) / len(self.episodes)

    @property
    def avg_invalid_actions(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.invalid_actions for e in self.episodes) / len(self.episodes)

    @property
    def total_wall_time(self) -> float:
        if self._start_time and self._end_time:
            return self._end_time - self._start_time
        return sum(e.wall_time_seconds for e in self.episodes)

    @property
    def error_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.error) / len(self.episodes)

    # ------------------------------------------------------------------
    # Per-category breakdown
    # ------------------------------------------------------------------

    def category_metrics(self) -> Dict[str, TaskMetrics]:
        """Aggregate metrics by app category."""
        cats: Dict[str, TaskMetrics] = {}
        for ep in self.episodes:
            cat = get_app_category(ep.task_name)
            if cat not in cats:
                cats[cat] = TaskMetrics(task_name=cat)
            cats[cat].add_episode(ep)
        return cats

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a rich text summary of evaluation results."""
        lines = []
        lines.append("=" * 70)
        lines.append(f"  ANDROID WORLD EVALUATION RESULTS — {self.split.upper()}")
        lines.append("=" * 70)
        lines.append("")

        # Aggregate
        lines.append("── Aggregate Metrics ──")
        lines.append(f"  Total episodes:        {self.n_episodes}")
        lines.append(f"  Unique task types:     {self.n_tasks}")
        lines.append(f"  Success rate:          {self.success_rate:.1%}")
        lines.append(f"  Avg steps (all):       {self.avg_steps:.2f}")
        lines.append(f"  Avg steps (success):   {self.avg_steps_on_success:.2f}")
        lines.append(f"  Median steps (success):{self.median_steps_on_success:.1f}")
        lines.append(f"  Avg reward:            {self.avg_reward:.3f}")
        lines.append(f"  Timeout rate:          {self.timeout_rate:.1%}")
        lines.append(f"  Avg invalid actions:   {self.avg_invalid_actions:.2f}")
        lines.append(f"  Error rate:            {self.error_rate:.1%}")
        lines.append(f"  Total wall time:       {self.total_wall_time:.1f}s")
        lines.append("")

        # Per-category
        cats = self.category_metrics()
        if cats:
            lines.append("── Per-Category Breakdown ──")
            lines.append(
                f"  {'Category':<16s} {'Episodes':>8s} {'SR':>8s} {'AvgSteps':>8s} {'Timeout':>8s} {'AvgRwd':>8s}"
            )
            lines.append("  " + "-" * 64)
            for cat_name in sorted(cats.keys()):
                m = cats[cat_name]
                lines.append(
                    f"  {cat_name:<16s} {m.n_episodes:>8d} {m.success_rate:>7.1%} "
                    f"{m.avg_steps:>8.1f} {m.timeout_rate:>7.1%} {m.avg_reward:>8.3f}"
                )
            lines.append("")

        # Per-task top/bottom
        if self.task_metrics:
            sorted_tasks = sorted(
                self.task_metrics.values(),
                key=lambda t: t.success_rate,
                reverse=True,
            )
            lines.append("── Per-Task Success Rates ──")
            lines.append(
                f"  {'Task':<45s} {'N':>4s} {'SR':>8s} {'AvgSteps':>8s}"
            )
            lines.append("  " + "-" * 68)
            for t in sorted_tasks:
                lines.append(
                    f"  {t.task_name:<45s} {t.n_episodes:>4d} "
                    f"{t.success_rate:>7.1%} {t.avg_steps:>8.1f}"
                )
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize results to a JSON-friendly dict."""
        return {
            "split": self.split,
            "aggregate": {
                "n_episodes": self.n_episodes,
                "n_tasks": self.n_tasks,
                "success_rate": self.success_rate,
                "avg_steps": self.avg_steps,
                "avg_steps_on_success": self.avg_steps_on_success,
                "median_steps_on_success": self.median_steps_on_success,
                "avg_reward": self.avg_reward,
                "timeout_rate": self.timeout_rate,
                "avg_invalid_actions": self.avg_invalid_actions,
                "error_rate": self.error_rate,
                "total_wall_time": self.total_wall_time,
            },
            "per_task": {
                name: {
                    "n_episodes": m.n_episodes,
                    "success_rate": m.success_rate,
                    "avg_steps": m.avg_steps,
                    "avg_steps_on_success": m.avg_steps_on_success,
                    "avg_reward": m.avg_reward,
                    "timeout_rate": m.timeout_rate,
                    "avg_invalid_actions": m.avg_invalid_actions,
                }
                for name, m in self.task_metrics.items()
            },
            "per_category": {
                name: {
                    "n_episodes": m.n_episodes,
                    "success_rate": m.success_rate,
                    "avg_steps": m.avg_steps,
                    "avg_reward": m.avg_reward,
                }
                for name, m in self.category_metrics().items()
            },
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def save(self, output_dir: str, filename: Optional[str] = None):
        """Save results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        fname = filename or f"eval_{self.split}_{int(time.time())}.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved eval results to {path}")
        return path

    @classmethod
    def from_episode_logs(
        cls, logs: List[Dict[str, Any]], split: str = "unknown"
    ) -> "EvalResults":
        """Reconstruct EvalResults from a list of episode log dicts."""
        results = cls(split=split)
        for log in logs:
            ep = EpisodeResult(
                task_name=log["task_name"],
                instance_id=log.get("instance_id", 0),
                success=log["success"],
                total_steps=log["total_steps"],
                total_reward=log.get("total_reward", 0.0),
                timeout=log.get("timeout", False),
                invalid_actions=log.get("invalid_actions", 0),
                wall_time_seconds=log.get("wall_time_seconds", 0.0),
                seed=log.get("seed"),
                error=log.get("error"),
            )
            results.add_episode(ep)
        return results

    @classmethod
    def from_json(cls, path: str) -> "EvalResults":
        """Load from a saved JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_episode_logs(data["episodes"], split=data.get("split", "unknown"))


# ============================================================================
# Evaluator — runs live evaluation episodes
# ============================================================================


class AndroidWorldEvaluator:
    """Runs evaluation episodes on AndroidWorld and collects metrics.

    This evaluator operates a provided AndroidWorldGame instance,
    stepping through episodes using a provided agent function.
    """

    def __init__(
        self,
        game: Union["AndroidWorldGame", List["AndroidWorldGame"]],
        task_set_config: "TaskSetConfig",
        split: str = "test_id",
        agent_fn: Any = None,
        n_instances_per_task: Optional[int] = None,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
        output_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Args:
            game: An initialized AndroidWorldGame instance, or a list of instances
                  for parallel evaluation across multiple emulators.
            task_set_config: TaskSetConfig with split definitions.
            split: Which split to evaluate ("test_id" or "test_ood").
            agent_fn: Callable(observation: str) -> str that returns the agent action.
                      If None, a dummy agent that always declares 'complete' is used.
            n_instances_per_task: Override config's n_instances_per_task.
            max_steps: Override config's eval_max_steps.
            seed: Override config's eval_seed.
            output_dir: Directory to save results.
            verbose: Print progress during evaluation.
        """
        if isinstance(game, list):
            self.games = game
        else:
            self.games = [game]
        self.game = self.games[0]  # backward compat
        self.num_workers = len(self.games)
        self.config = task_set_config
        self.split = split
        self.agent_fn = agent_fn or self._dummy_agent
        self.n_instances = n_instances_per_task or task_set_config.n_instances_per_task
        self.max_steps = max_steps or task_set_config.eval_max_steps
        self.seed = seed if seed is not None else task_set_config.eval_seed
        self.output_dir = output_dir
        self.verbose = verbose

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    def _print_progress_header(self, n_tasks: int, total: int):
        """Print the evaluation progress header."""
        print(flush=True)
        print("┌" + "─" * 78 + "┐", flush=True)
        print(f"│  Eval: {self.split.upper():<12s}  "
              f"Tasks: {n_tasks}  "
              f"Episodes: {total}  "
              f"Max steps: {self.max_steps:<6d}"
              f"{'':>10s}│", flush=True)
        print("├" + "─" * 78 + "┤", flush=True)
        print(f"│  {'#':>4s}  {'Task':<35s} {'Result':>6s} "
              f"{'Steps':>5s} {'Rwd':>6s} {'Time':>6s} {'SR':>8s} │", flush=True)
        print("├" + "─" * 78 + "┤", flush=True)

    def _print_progress_row(self, idx: int, total: int, ep: EpisodeResult,
                            n_success: int, n_done: int):
        """Print a single progress row for a completed episode."""
        status = "✓ OK" if ep.success else ("⏱ TMO" if ep.timeout else ("⚠ ERR" if ep.error else "✗ FAIL"))
        sr = n_success / n_done if n_done > 0 else 0.0
        task_display = ep.task_name
        if len(task_display) > 35:
            task_display = task_display[:32] + "..."
        time_str = self._format_duration(ep.wall_time_seconds)
        print(f"│  {idx:>4d}/{total:<4d} {task_display:<35s} {status:>6s} "
              f"{ep.total_steps:>5d} {ep.total_reward:>+6.1f} {time_str:>6s} "
              f"{sr:>7.1%} │", flush=True)

    def _print_progress_footer(self, results: EvalResults):
        """Print the evaluation progress footer with final stats."""
        elapsed = self._format_duration(results.total_wall_time)
        print("├" + "─" * 78 + "┤", flush=True)
        print(f"│  {'DONE':>4s}  "
              f"Success: {results.success_rate:.1%}  "
              f"Avg steps: {results.avg_steps:.1f}  "
              f"Avg reward: {results.avg_reward:+.2f}  "
              f"Elapsed: {elapsed:<8s}"
              f"{'':>4s}│", flush=True)
        print("└" + "─" * 78 + "┘", flush=True)
        print(flush=True)

    def run(self) -> EvalResults:
        """Run all evaluation episodes and return results.

        When multiple game instances are available (num_workers > 1),
        episodes are distributed across emulators in parallel using a
        thread pool with a game-pool pattern.
        """
        if self.num_workers > 1:
            return self._run_parallel()
        return self._run_sequential()

    def _run_sequential(self) -> EvalResults:
        """Run all evaluation episodes sequentially on a single game."""
        tasks = self.config.get_tasks(self.split)
        results = EvalResults(split=self.split)
        results._start_time = time.time()

        total = len(tasks) * self.n_instances
        completed = 0
        n_success = 0

        rng = random.Random(self.seed) if self.seed is not None else random.Random()

        if self.verbose:
            self._print_progress_header(len(tasks), total)

        for task_name in tasks:
            for instance_id in range(self.n_instances):
                ep_seed = rng.randint(0, 2**31) if self.seed is not None else None

                completed += 1
                if self.verbose:
                    task_display = task_name if len(task_name) <= 35 else task_name[:32] + "..."
                    print(f"│  {completed:>4d}/{total:<4d} {task_display:<35s} {'...':>6s} "
                          f"{'':>5s} {'':>6s} {'':>6s} {'':>8s}│",
                          end="\r", flush=True)

                ep = self._run_episode_on_game(self.game, task_name, instance_id, ep_seed)
                results.add_episode(ep)
                n_success += int(ep.success)

                if self.verbose:
                    self._print_progress_row(completed, total, ep, n_success, completed)

        results._end_time = time.time()

        if self.verbose:
            self._print_progress_footer(results)

        if self.output_dir:
            results.save(self.output_dir)

        if self.verbose:
            print(results.summary())

        return results

    def _run_parallel(self) -> EvalResults:
        """Run evaluation episodes in parallel across multiple game instances."""
        tasks = self.config.get_tasks(self.split)
        results = EvalResults(split=self.split)
        results._start_time = time.time()

        # Pre-compute all work items with deterministic seeds
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        work_items = []
        for task_name in tasks:
            for instance_id in range(self.n_instances):
                ep_seed = rng.randint(0, 2**31) if self.seed is not None else None
                work_items.append((task_name, instance_id, ep_seed))

        total = len(work_items)
        completed = 0
        n_success = 0
        results_lock = threading.Lock()

        if self.verbose:
            self._print_progress_header(len(tasks), total)
            print(f"│  {'':>4s}  Using {self.num_workers} emulators in parallel"
                  f"{'':>35s}│", flush=True)
            print("├" + "─" * 78 + "┤", flush=True)

        # Create game pool queue
        game_pool: queue.Queue = queue.Queue()
        for g in self.games:
            game_pool.put(g)

        def _worker(task_name: str, instance_id: int, ep_seed: Optional[int]) -> EpisodeResult:
            nonlocal completed, n_success
            game = game_pool.get()
            try:
                ep = self._run_episode_on_game(game, task_name, instance_id, ep_seed)
            finally:
                game_pool.put(game)

            with results_lock:
                results.add_episode(ep)
                completed += 1
                n_success += int(ep.success)
                if self.verbose:
                    self._print_progress_row(completed, total, ep, n_success, completed)
            return ep

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(_worker, task_name, instance_id, ep_seed)
                for task_name, instance_id, ep_seed in work_items
            ]
            # Wait for all futures to complete (results already collected via lock)
            for f in futures:
                f.result()  # raises if worker raised

        results._end_time = time.time()

        if self.verbose:
            self._print_progress_footer(results)

        if self.output_dir:
            results.save(self.output_dir)

        if self.verbose:
            print(results.summary())

        return results

    def _run_episode_on_game(
        self,
        game: "AndroidWorldGame",
        task_name: str,
        instance_id: int,
        seed: Optional[int],
    ) -> EpisodeResult:
        """Run a single evaluation episode on a specific game instance."""
        ep_start = time.time()
        total_reward = 0.0
        invalid_actions = 0
        error_msg = None
        success = False
        timeout = False
        steps = 0

        try:
            # Reset game for this task
            obs = game.reset(task_type=task_name, seed=seed)

            for step_i in range(self.max_steps):
                # Agent produces action
                action = self.agent_fn(obs)
                result = game.step(action)

                steps += 1
                total_reward += result.reward

                # Track invalid actions
                if result.reward == game.REWARD_INVALID_ACTION:
                    invalid_actions += 1

                obs = result.observation

                if result.done:
                    success = result.reward >= game.REWARD_SUCCESS
                    timeout = False
                    break
            else:
                # Exhausted max_steps without done signal
                timeout = True

        except Exception as e:
            logger.error(f"Episode error ({task_name}): {e}", exc_info=True)
            error_msg = str(e)

        return EpisodeResult(
            task_name=task_name,
            instance_id=instance_id,
            success=success,
            total_steps=steps,
            total_reward=total_reward,
            timeout=timeout,
            invalid_actions=invalid_actions,
            wall_time_seconds=time.time() - ep_start,
            seed=seed,
            error=error_msg,
        )

    @staticmethod
    def _dummy_agent(obs: str) -> str:
        """Dummy agent that always declares the task complete."""
        return 'Reason: Declaring complete.\nAction: {"action_type": "status", "goal_status": "complete"}'
