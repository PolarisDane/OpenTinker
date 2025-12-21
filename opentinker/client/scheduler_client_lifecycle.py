#!/usr/bin/env python3
"""
Scheduler Client Lifecycle Management

This module provides lifecycle management utilities for scheduler-based clients,
including:
- Automatic reward server startup and shutdown
- Job cleanup on exit (both normal and interrupted)
- Signal handling for graceful shutdown
"""

import atexit
import signal
import sys
import subprocess
import os
import time
import requests
from typing import Optional, Callable


class SchedulerClientLifecycleManager:
    """
    Manages the lifecycle of scheduler-based training clients.
    
    Handles:
    - Reward server process management
    - Job cleanup on exit
    - Signal handling (SIGINT, SIGTERM)
    """
    
    def __init__(self):
        self._reward_server_process: Optional[subprocess.Popen] = None
        self._scheduler_client = None
        self._current_job_id: Optional[str] = None
        self._cleanup_callbacks: list[Callable] = []
        self._signal_handlers_registered = False
        
    def start_reward_server(
        self, 
        reward_server_path: str = "../reward_functions/math_reward_server.py", 
        reward_ip: str = "localhost", 
        reward_port: Optional[int] = None, 
        wait_time: int = 5
    ) -> tuple[subprocess.Popen, int]:
        """
        Start the reward server in the background.
        
        Args:
            reward_server_path: Path to the reward server script (relative or absolute)
            reward_ip: IP address for the reward server (default: "localhost")
            reward_port: Port number for the reward server (default: None, auto-assigned)
            wait_time: Time to wait for server to start (default: 5 seconds)
        
        Returns:
            tuple: (subprocess.Popen, int) - The reward server process and the port number used
        """
        from utils import find_free_port
        
        # Auto-assign port if not specified
        if reward_port is None:
            reward_port = find_free_port()
        
        # Resolve the path to the reward server
        if not os.path.isabs(reward_server_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            reward_server_path = os.path.normpath(os.path.join(current_dir, reward_server_path))
        
        if not os.path.exists(reward_server_path):
            raise FileNotFoundError(f"Reward server not found at: {reward_server_path}")
        
        print(f"Starting reward server from: {reward_server_path}")
        print(f"Server will listen on: http://{reward_ip}:{reward_port}")
        
        # Start the server process with port argument
        self._reward_server_process = subprocess.Popen(
            ["python", reward_server_path, "--port", str(reward_port), "--host", reward_ip],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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
        
        return self._reward_server_process, reward_port
    
    def register_job(self, scheduler_client, job_id: str):
        """
        Register a job for cleanup on exit.
        
        Args:
            scheduler_client: The scheduler client instance
            job_id: The job ID to track
        """
        self._scheduler_client = scheduler_client
        self._current_job_id = job_id
    
    def add_cleanup_callback(self, callback: Callable):
        """
        Add a custom cleanup callback to be called on exit.
        
        Args:
            callback: A callable that takes no arguments
        """
        self._cleanup_callbacks.append(callback)
    
    def _cleanup_reward_server(self):
        """Clean up the reward server process."""
        if self._reward_server_process:
            print("\nShutting down reward server...")
            self._reward_server_process.terminate()
            try:
                self._reward_server_process.wait(timeout=5)
                print("✓ Reward server stopped")
            except subprocess.TimeoutExpired:
                print("⚠ Force killing reward server...")
                self._reward_server_process.kill()
            self._reward_server_process = None
    
    def _cleanup_job(self):
        """Clean up job from scheduler."""
        if self._scheduler_client and self._current_job_id:
            try:
                print(f"\n🧹 Cleaning up job {self._current_job_id}...")
                # Try to cancel the job first (works for QUEUED, STARTING, and RUNNING jobs)
                try:
                    result = self._scheduler_client.cancel_job(self._current_job_id)
                    print(f"✓ Job {self._current_job_id} cancelled: {result.get('message', 'Success')}")
                except Exception as cancel_error:
                    # If cancel fails (e.g., job already completed), try to complete it
                    print(f"⚠ Could not cancel job (might be already completed): {cancel_error}")
                    try:
                        self._scheduler_client.complete_job(self._current_job_id)
                        print(f"✓ Job {self._current_job_id} marked as complete")
                    except Exception as complete_error:
                        print(f"⚠ Could not complete job: {complete_error}")
            except Exception as e:
                print(f"⚠ Failed to clean up job: {e}")
            self._current_job_id = None
    
    def _cleanup_all(self):
        """Run all cleanup operations."""
        # Run custom callbacks first
        for callback in self._cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"⚠ Cleanup callback failed: {e}")
        
        # Then clean up job and reward server
        self._cleanup_job()
        self._cleanup_reward_server()
    
    def _signal_handler(self, signum, frame):
        """Handle SIGINT and SIGTERM for graceful shutdown."""
        signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
        print(f"\n\n{'='*60}")
        print(f"⚠️  Received {signal_name} - Initiating graceful shutdown")
        print(f"{'='*60}")
        
        self._cleanup_all()
        
        print(f"\n👋 Shutdown complete. Exiting...\n")
        sys.exit(0)
    
    def enable_auto_cleanup(self):
        """
        Enable automatic cleanup on exit and signal handling.
        
        This registers:
        - atexit handlers for normal exit
        - signal handlers for SIGINT (Ctrl+C) and SIGTERM
        """
        if not self._signal_handlers_registered:
            # Register cleanup functions with atexit
            atexit.register(self._cleanup_all)
            
            # Register signal handlers
            signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl+C
            signal.signal(signal.SIGTERM, self._signal_handler)  # kill command
            
            self._signal_handlers_registered = True


# Global lifecycle manager instance
_global_lifecycle_manager: Optional[SchedulerClientLifecycleManager] = None


def get_lifecycle_manager() -> SchedulerClientLifecycleManager:
    """
    Get or create the global lifecycle manager instance.
    
    Returns:
        SchedulerClientLifecycleManager: The global lifecycle manager
    """
    global _global_lifecycle_manager
    if _global_lifecycle_manager is None:
        _global_lifecycle_manager = SchedulerClientLifecycleManager()
        _global_lifecycle_manager.enable_auto_cleanup()
    return _global_lifecycle_manager
