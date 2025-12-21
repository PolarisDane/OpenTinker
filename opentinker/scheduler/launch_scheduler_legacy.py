#!/usr/bin/env python3
"""
Launch Job Scheduler for OpenTinker Training

This script starts the job scheduler server that manages training jobs
across multiple GPU resources.

Example usage:
    python launch_scheduler.py \
        available_gpus=[0,1,2,3] \
        port_range=[38564,38600] \
        scheduler_port=8765
"""

import hydra
import logging
import os
import ray
import uvicorn
import signal
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from job_scheduler import JobSchedulerActor, create_app
from user_management import UserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global variables for cleanup
scheduler_actor_instance = None


def cleanup_scheduler():
    """Clean up scheduler resources on shutdown"""
    global scheduler_actor_instance
    
    logger.info("\n" + "="*60)
    logger.info("🧹 Cleaning up scheduler resources...")
    logger.info("="*60)
    
    try:
        if scheduler_actor_instance:
            logger.info("Shutting down scheduler actor...")
            try:
                # Try to kill the actor gracefully with timeout
                # Don't wait for the kill to complete - just send the signal
                ray.kill(scheduler_actor_instance, no_restart=True)
                logger.info("✓ Scheduler actor kill signal sent")
            except Exception as e:
                # If kill fails, log but continue cleanup
                logger.warning(f"Failed to kill scheduler actor (it may already be dead): {e}")
    except Exception as e:
        logger.error(f"Error during scheduler actor cleanup: {e}")
    
    try:
        if ray.is_initialized():
            logger.info("Shutting down Ray...")
            # Force immediate shutdown without waiting
            ray.shutdown()
            logger.info("✓ Ray shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down Ray: {e}")
    
    logger.info("="*60)
    logger.info("👋 Scheduler cleanup complete")
    logger.info("="*60 + "\n")


def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM for graceful shutdown"""
    signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
    logger.info(f"\n\n{'='*60}")
    logger.info(f"⚠️ Received {signal_name} - Initiating graceful shutdown")
    logger.info(f"{'='*60}\n")
    
    cleanup_scheduler()
    
    logger.info("Exiting scheduler...\n")
    sys.exit(0)


@hydra.main(config_path="config", config_name="scheduler", version_base=None)
def main(cfg: DictConfig):
    """
    Launch the job scheduler.
    
    Args:
        cfg: Hydra configuration
    """
    logger.info("=" * 60)
    logger.info("OpenTinker Job Scheduler")
    logger.info("=" * 60)
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        logger.info("Initializing Ray...")
        ray.init(
            ignore_reinit_error=True,
            logging_level=logging.INFO,
        )
        logger.info("Ray initialized successfully")
    else:
        logger.info("Ray already initialized")
    
    # Parse configuration
    available_gpus = list(cfg.available_gpus)
    scheduler_port = cfg.scheduler_port
    enable_auth = cfg.get("enable_auth", True)
    db_path = cfg.get("user_db_path", "scheduler_users.db")
    gpus_per_job = cfg.get("gpus_per_job", 4)  # Default to 4 GPUs per job
    
    # Port range is optional - if not provided, auto-detect
    port_range = None
    num_ports = cfg.get("num_ports", 50)  # Default to 50 ports
    
    if "port_range" in cfg and cfg.port_range is not None:
        port_range = (cfg.port_range[0], cfg.port_range[1])
        logger.info(f"Using manual port range: {port_range}")
    else:
        logger.info(f"Port range not specified, will auto-detect {num_ports} available ports")
    
    # Get paths
    base_dir = Path(__file__).parent.parent.parent.parent.absolute()
    server_script_path = base_dir / "opentinker/server/launch_http_server.py"
    
    if not server_script_path.exists():
        raise FileNotFoundError(f"Server script not found: {server_script_path}")
    
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"GPUs per job: {gpus_per_job}")
    logger.info(f"Scheduler port: {scheduler_port}")
    logger.info(f"Authentication: {'enabled' if enable_auth else 'disabled'}")
    logger.info(f"User database: {db_path}")
    logger.info(f"Server script: {server_script_path}")
    logger.info(f"Base directory: {base_dir}")
    
    # Initialize UserManager
    logger.info("Initializing user management...")
    user_manager = UserManager(db_path=db_path)
    
    # Create default admin user if it doesn't exist
    admin_user = user_manager.create_default_admin()
    if admin_user:
        logger.info("=" * 60)
        logger.info("🔑 DEFAULT ADMIN CREDENTIALS")
        logger.info("=" * 60)
        logger.info(f"Username: {admin_user.username}")
        logger.info(f"API Key:  {admin_user.api_key}")
        logger.info("=" * 60)
        logger.info("⚠️  SAVE THIS API KEY - IT CANNOT BE RETRIEVED LATER!")
        logger.info("=" * 60)
    
    # Get logs directory from config or use default
    logs_dir = cfg.get("logs_dir", "/workspace/logs")
    logger.info(f"Job logs directory: {logs_dir}")
    
    # Create scheduler actor
    logger.info("Creating scheduler actor...")
    scheduler_actor = JobSchedulerActor.remote(
        available_gpus=available_gpus,
        port_range=port_range,
        server_script_path=str(server_script_path),
        base_dir=str(base_dir),
        num_ports=num_ports,
        gpus_per_job=gpus_per_job,
        logs_dir=logs_dir,
    )
    logger.info("Scheduler actor created")
    
    # Store globally for signal handler
    global scheduler_actor_instance
    scheduler_actor_instance = scheduler_actor
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command
    logger.info("Signal handlers registered for graceful shutdown")
    
    # Create FastAPI app with authentication
    app = create_app(scheduler_actor, user_manager, enable_auth=enable_auth)
    
    # Run server
    logger.info("=" * 60)
    logger.info(f"Starting scheduler server on port {scheduler_port}")
    logger.info(f"Access API docs at: http://localhost:{scheduler_port}/docs")
    if enable_auth:
        logger.info("Authentication is ENABLED - API key required for all operations")
        logger.info("Register users at: POST /register?username=<username>")
    else:
        logger.info("Authentication is DISABLED - no API key required")
    logger.info("=" * 60)
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=scheduler_port,
            log_level="info",
        )
    except KeyboardInterrupt:
        # This should be caught by signal handler, but just in case
        logger.info("\nKeyboardInterrupt detected")
        cleanup_scheduler()
    finally:
        # Ensure cleanup happens
        if ray.is_initialized():
            cleanup_scheduler()


if __name__ == "__main__":
    main()
