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

import asyncio
import hydra
import logging
import ray
import signal
import uvicorn
from pathlib import Path
from omegaconf import DictConfig

from job_scheduler import JobSchedulerActor, create_app
from user_management import UserManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for cleanup
scheduler_actor_instance = None
_cleanup_done = False


def cleanup_scheduler():
    """Clean up scheduler resources on shutdown."""
    global scheduler_actor_instance, _cleanup_done

    if _cleanup_done:
        return
    _cleanup_done = True

    logger.info("\n" + "=" * 60)
    logger.info("🧹 Cleaning up scheduler resources...")
    logger.info("=" * 60)

    try:
        if scheduler_actor_instance is not None:
            logger.info("Shutting down scheduler actor...")
            try:
                if ray.is_initialized():
                    ray.kill(scheduler_actor_instance, no_restart=True)
                    logger.info("✓ Scheduler actor kill signal sent")
                else:
                    logger.warning("Ray not initialized. Skipping ray.kill on actor.")
            except Exception as e:
                logger.warning(
                    f"Failed to kill scheduler actor (it may already be dead): {e}"
                )
    except Exception as e:
        logger.error(f"Error during scheduler actor cleanup: {e}")

    try:
        if ray.is_initialized():
            logger.info("Shutting down Ray...")
            ray.shutdown()
            logger.info("✓ Ray shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down Ray: {e}")

    logger.info("=" * 60)
    logger.info("👋 Scheduler cleanup complete")
    logger.info("=" * 60 + "\n")


async def serve_uvicorn(app, scheduler_port: int):
    """
    Run uvicorn server with explicit lifecycle control.
    This makes Ctrl+C and SIGTERM reliable with Ray in the same process.
    """
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=int(scheduler_port),
        log_level="info",
        loop="asyncio",
        lifespan="on",
    )
    server = uvicorn.Server(config)

    # Signal handling. Prefer loop.add_signal_handler on Unix.
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_shutdown(sig_name: str):
        logger.info("\n" + "=" * 60)
        logger.info(f"⚠️ Received {sig_name}. Initiating graceful shutdown")
        logger.info("=" * 60 + "\n")
        try:
            server.should_exit = True
        finally:
            stop_event.set()

    for sig, name in ((signal.SIGINT, "SIGINT"), (signal.SIGTERM, "SIGTERM")):
        try:
            loop.add_signal_handler(sig, _request_shutdown, name)
        except (NotImplementedError, RuntimeError):
            # Fallback. Windows and some environments do not support add_signal_handler.
            signal.signal(sig, lambda s, f, _name=name: _request_shutdown(_name))

    serve_task = asyncio.create_task(server.serve())

    # Wait until shutdown is requested or the server exits by itself.
    done, pending = await asyncio.wait(
        {serve_task, asyncio.create_task(stop_event.wait())},
        return_when=asyncio.FIRST_COMPLETED,
    )

    # If the stop event fired first, ensure the server gets a chance to stop.
    if serve_task not in done:
        server.should_exit = True
        await serve_task

    # Cancel any leftover task (the stop_event waiter).
    for t in pending:
        t.cancel()
        try:
            await t
        except Exception:
            pass


async def main_async(cfg: DictConfig):
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
    scheduler_port = int(cfg.scheduler_port)
    enable_auth = bool(cfg.get("enable_auth", True))
    db_path = cfg.get("user_db_path", "scheduler_users.db")
    gpus_per_job = int(cfg.get("gpus_per_job", 4))
    num_ports = int(cfg.get("num_ports", 50))

    port_range = None
    if "port_range" in cfg and cfg.port_range is not None:
        port_range = (int(cfg.port_range[0]), int(cfg.port_range[1]))
        logger.info(f"Using manual port range: {port_range}")
    else:
        logger.info(
            f"Port range not specified, will auto-detect {num_ports} available ports"
        )

    # Get paths
    base_dir = Path(__file__).parent.parent.parent.absolute()
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
        logger.info("⚠️  SAVE THIS API KEY. IT CANNOT BE RETRIEVED LATER!")
        logger.info("=" * 60)

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

    global scheduler_actor_instance
    scheduler_actor_instance = scheduler_actor

    # Create FastAPI app with authentication
    app = create_app(scheduler_actor, user_manager, enable_auth=enable_auth)

    # Run server
    logger.info("=" * 60)
    logger.info(f"Starting scheduler server on port {scheduler_port}")
    logger.info(f"Access API docs at: http://localhost:{scheduler_port}/docs")
    if enable_auth:
        logger.info("Authentication is ENABLED. API key required for all operations")
        logger.info("Register users at: POST /register?username=<username>")
    else:
        logger.info("Authentication is DISABLED. No API key required")
    logger.info("=" * 60)

    try:
        await serve_uvicorn(app, scheduler_port)
    finally:
        cleanup_scheduler()


@hydra.main(config_path="config", config_name="scheduler", version_base=None)
def main(cfg: DictConfig):
    try:
        asyncio.run(main_async(cfg))
    except KeyboardInterrupt:
        # Should be handled by our signal path, but keep as a last resort.
        logger.info("KeyboardInterrupt detected at top-level. Cleaning up.")
        cleanup_scheduler()
    except Exception:
        # Ensure cleanup on unexpected crashes too.
        cleanup_scheduler()
        raise


if __name__ == "__main__":
    main()
