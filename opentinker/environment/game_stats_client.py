#!/usr/bin/env python3
"""Game Server Stats Client.

Helper class to query metrics from game server during training.
Supports multi-job statistics isolation via job_id parameter.

Usage:
    from game_stats_client import GameStatsClient
    
    # Single job (backward compatible)
    stats_client = GameStatsClient("http://localhost:8081")
    
    # Multi-job with isolation
    stats_client = GameStatsClient("http://localhost:8081", job_id="training_job_1")
    
    # Before each training step
    stats_client.reset_step()
    
    # ... training step runs, game server records stats ...
    
    # After each training step
    metrics = stats_client.get_step_stats()
    print(f"Win rate: {metrics['win_rate']:.2%}")
"""

import requests
from typing import Any, Dict, Optional


class GameStatsClient:
    """Client for querying game server statistics.
    
    Supports multi-job statistics isolation via job_id parameter.
    """
    
    def __init__(self, game_server_url: str, job_id: str = "default", timeout: float = 5.0):
        """Initialize stats client.
        
        Args:
            game_server_url: URL of the game server (e.g., http://localhost:8081)
            job_id: Job identifier for statistics isolation (default: "default")
            timeout: Request timeout
        """
        self.url = game_server_url.rstrip("/")
        self.job_id = job_id
        self.timeout = timeout
    
    def reset_step(self) -> Dict[str, Any]:
        """Reset step statistics. Call at the START of each training step.
        
        Returns:
            Response with new step number
        """
        response = requests.post(
            f"{self.url}/stats/reset",
            params={"job_id": self.job_id},
            json={},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_step_stats(self) -> Dict[str, Any]:
        """Get statistics for the current step.
        
        Returns:
            Dict with:
            - step: Current step number
            - win_rate: Win rate for this step
            - loss_rate: Loss rate for this step
            - draw_rate: Draw rate for this step
            - timeout_rate: Timeout rate for this step
            - games_in_step: Number of completed games
            - mean_reward: Mean reward for all interactions
            - job_id: The job ID for which stats are returned
        """
        response = requests.get(
            f"{self.url}/stats/step",
            params={"job_id": self.job_id},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get both step and cumulative statistics.
        
        Returns:
            Dict with step stats + cumulative stats
        """
        response = requests.get(
            f"{self.url}/stats",
            params={"job_id": self.job_id},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_cumulative_stats(self) -> Dict[str, Any]:
        """Get cumulative statistics across all steps.
        
        Returns:
            Dict with total wins, losses, games, and cumulative rates
        """
        response = requests.get(
            f"{self.url}/stats/cumulative",
            params={"job_id": self.job_id},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def reset_all(self) -> Dict[str, Any]:
        """Reset all statistics for this job. Call when starting a new training run.
        
        Returns:
            Confirmation message
        """
        response = requests.post(
            f"{self.url}/stats/reset_all",
            params={"job_id": self.job_id},
            json={},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def list_jobs(self) -> Dict[str, Any]:
        """List all active job IDs on the server.
        
        Returns:
            Dict with 'jobs' list and 'count'
        """
        response = requests.get(
            f"{self.url}/stats/jobs",
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> bool:
        """Check if game server is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception:
            return False

