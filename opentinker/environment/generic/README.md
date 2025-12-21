# Generic Environment for LLM-Environment Interaction

This directory contains the generic environment implementation for training LLMs
to interact with external environments (like OpenAI Gym).

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐        ┌─────────────────────────────────────┐  │
│  │ GenericEnvironment│───────▶│       GenericAgentLoop              │  │
│  │  (BaseEnvironment) │        │  (verl/experimental/agent_loop/)   │  │
│  │                   │        │                                     │  │
│  │ - Dataloader     │        │  PENDING → GENERATING → INTERACTING │  │
│  │ - InteractionSpec │        │              │              │       │  │
│  └─────────────────┘        │              ▼              ▼       │  │
│                              │        LLM Server    Environment   │  │
│                              │        (mask=1)      (mask=0)      │  │
│                              └─────────────────────────────────────┘  │
│                                              │                         │
│                              ┌───────────────┴───────────────┐        │
│                              │    BaseInteraction            │        │
│                              │  (verl/interactions/)          │        │
│                              │                               │        │
│                              │  - GymEnvironmentInteraction  │        │
│                              │  - SimpleTextEnvironment      │        │
│                              │  - Gsm8kInteraction           │        │
│                              └───────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Concept: Environment Provides Rewards

Unlike standard PPO training where a separate reward function evaluates completions,
in environment interaction:

- **Reward comes from the environment** via `interaction.generate_response()`
- **No external `reward_function` is needed**
- `response_mask` ensures only LLM tokens contribute to the loss

## Quick Start

```python
from omegaconf import OmegaConf
from opentinker.environment.generic.generic_env import (
    GenericEnvironment,
    InteractionSpec,
)

# 1. Configure environment
config = OmegaConf.create({
    "tokenizer_path": "meta-llama/Llama-2-7b-chat-hf",
    "data_path": "data/train.parquet",
    "max_prompt_tokens": 1024,
    "max_new_tokens": 512,
    "batch_size": 4,
    "num_workers": 4,
    "algorithm": "agent_loop",
})

# 2. Define interaction with Gym environment
interaction_specs = [
    InteractionSpec(
        name="my_env",
        class_path="verl.interactions.gym_environment_interaction.GymEnvironmentInteraction",
        config={"env_endpoint": "http://localhost:8080", "max_steps": 100}
    )
]

# 3. Create environment
env = GenericEnvironment(config, interaction_specs)

# 4. Use with training client
train_dl, val_dl = env.get_dataloader()
env_config = env.setup(client)
```

## Files

| File | Description |
|------|-------------|
| `generic_env.py` | Main GenericEnvironment class |
| `example_usage.py` | Usage examples |

## Dataset Format

Your training data should include `interaction_kwargs` to specify which interaction to use:

```json
{
  "prompt": [
    {"role": "system", "content": "You are playing a text adventure..."},
    {"role": "user", "content": "You are in a cave. What do you do?"}
  ],
  "extra_info": {
    "interaction_kwargs": {"name": "my_env"}
  }
}
```

## Related Files

- Agent Loop: `verl/experimental/agent_loop/generic_agent_loop.py`
- Interactions: `verl/interactions/gym_environment_interaction.py`
- Mock Server: `opentinker/environment/example/mock_env_server.py`
