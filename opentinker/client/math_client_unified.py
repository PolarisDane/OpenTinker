#!/usr/bin/env python3
"""
Example: Math Training with GameEnvironment Pattern

Uses the same GameEnvironment pattern as Gomoku with reward computed in game step().

Usage:
    1. Start the Math game server:
       python opentinker/environment/math/math_server.py
       
    2. Start the scheduler:
       python opentinker/scheduler/launch_scheduler.py
       
    3. Run this client:
       python math_client_unified.py
"""

import hydra
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader

from http_training_client import ServiceClient, SchedulerClient
from opentinker.environment.base_game_environment import GameEnvironment
from opentinker.environment.base_data_generator import DynamicGameDataset, collate_fn
from opentinker.environment.math import MathGame
from opentinker.environment.static_data_generator import StaticDatasetGenerator
from opentinker.environment.game_stats_client import GameStatsClient
from utils import resolve_paths_in_config
from scheduler_client_lifecycle import get_lifecycle_manager
from verl.trainer.main_ppo import create_rl_sampler

class MathGameEnvironment(GameEnvironment):
    """GameEnvironment for static dataset math problems."""
    
    def __init__(self, game_class, config, data_paths, val_data_paths=None, game_kwargs=None, job_id=None):
        self.data_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
        self.val_data_paths = [val_data_paths] if isinstance(val_data_paths, str) else (list(val_data_paths) if val_data_paths else None)
        super().__init__(game_class=game_class, config=config, game_kwargs=game_kwargs or {}, job_id=job_id)
    
    def _setup_dataloader(self):
        """Use StaticDatasetGenerator for static dataset."""
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        dataset_config = OmegaConf.create({
            "max_prompt_length": self.config.max_prompt_tokens,
            "truncation": "right",
            "return_raw_chat": True,
        })

        math_game_for_prompt = MathGame()

        
        # Training data generator
        train_generator = StaticDatasetGenerator(
            data_paths=self.data_paths,
            interaction_name=self.interaction_name,
            prompt_key="prompt",
            ground_truth_key="ground_truth",
            shuffle=True,
            system_prompt=math_game_for_prompt.get_system_prompt(),
        )
        
        batch_size = self.config.batch_size
        num_steps = getattr(self.config, 'num_steps', None)
        virtual_size = num_steps * batch_size if num_steps else len(train_generator) * getattr(self.config, 'num_epochs', 1)
        
        train_dataset = DynamicGameDataset(train_generator, tokenizer, dataset_config, virtual_size=virtual_size)

        sampler_config = OmegaConf.create({
            "shuffle": True,
            "seed": 42,
            "sampler": None,
        })
        train_sampler = create_rl_sampler(sampler_config, train_dataset)


        self.train_dataloader = StatefulDataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                                        sampler=train_sampler,
                                        num_workers=getattr(self.config, 'num_workers', 0),
                                        collate_fn=collate_fn, drop_last=True)
        print(f"Training dataloader: {len(self.train_dataloader)} batches")
        
        # Validation data generator - sample exactly val_batch_size samples, keep fixed
        if self.val_data_paths:
            val_generator = StaticDatasetGenerator(
                data_paths=self.val_data_paths,
                interaction_name=self.interaction_name,
                prompt_key="prompt",
                ground_truth_key="ground_truth",
                shuffle=False,  # No shuffle - keep samples fixed
                seed=42,
                system_prompt=math_game_for_prompt.get_system_prompt(),
            )
            val_batch_size = getattr(self.config, 'val_batch_size', min(64, len(val_generator)))
            # Use val_batch_size as virtual_size to sample exactly that many samples
            val_dataset = DynamicGameDataset(val_generator, tokenizer, dataset_config, 
                                             virtual_size=val_batch_size, seed=42)
            self.val_dataloader = StatefulDataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                                             num_workers=getattr(self.config, 'num_workers', 0),
                                             collate_fn=collate_fn, drop_last=False)
            print(f"Validation dataloader: {val_batch_size} fixed samples in {len(self.val_dataloader)} batch(es)")



@hydra.main(config_path="client_config", config_name="math_param.yaml")
def main(args):
    args = resolve_paths_in_config(args)
    lifecycle = get_lifecycle_manager()
    
    print("=" * 60)
    print("Math Training with GameEnvironment Pattern")
    print("=" * 60)
    
    # 1. Submit job to scheduler
    scheduler_client = SchedulerClient(
        scheduler_url=args.get("scheduler_url", "http://localhost:8780"),
        api_key=args.get("scheduler_api_key")
    )
    
    job_result = scheduler_client.submit_job(
        config=OmegaConf.to_container(args, resolve=True),
        enable_agent_loop=True,
        wandb_key=args.get("wandb_key"),
        num_gpus=args.get("num_gpus"),
    )
    
    job_id = job_result["job_id"]
    server_url = job_result["server_url"]
    lifecycle.register_job(scheduler_client, job_id)
    
    print(f"✓ Job {job_id} allocated at {server_url}")
    
    # 2. Setup environment (job_id is automatically handled)
    env_endpoint = args.interaction.config.env_endpoint
    env = MathGameEnvironment(
        game_class=MathGame,
        config=args,
        data_paths=[args.data_path],
        val_data_paths=[args.val_data_path] if args.val_data_path else None,
        job_id=job_id,  # Pass job_id directly
    )
    print(f"✓ Environment created, interaction config: {env.get_interaction_config_path()}")
    
    # 3. Setup game stats client (use env.job_id for consistency)
    game_stats = GameStatsClient(env_endpoint, job_id=env.job_id)
    if game_stats.health_check():
        game_stats.reset_all()
        print(f"✓ Connected to math server at {env_endpoint}")
    else:
        game_stats = None
        print(f"⚠ Math server not responding at {env_endpoint}")
    
    # 4. Connect to training server
    client = ServiceClient(
        server_url=server_url,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        logger_backends=args.logger_backends,
    )
    client.set_config(args, env)
    
    # 5. Train
    print(f"Starting training: steps={args.get('num_steps')}, epochs={args.get('num_epochs')}")
    
    try:
        final_metrics = client.fit(
            env=env,
            num_epochs=args.get("num_epochs"),
            num_steps=args.get("num_steps"),
            save_freq=args.save_freq,
            test_freq=args.test_freq,
            verbose=True,
            validate_before_training=True,
            game_stats_client=game_stats,
        )
        print(f"Training completed! Metrics: {final_metrics}")
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
