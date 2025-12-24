# Geo3K Vision-Language Training Quick Start

## Overview

This guide shows how to train vision-language models (like Qwen2.5-VL) on Geo3K geometry problems using OpenTinker.

## Prerequisites

```bash
# Install required packages
pip install transformers>=4.37.0 pillow

# Prepare Geo3K data (if not already done)
cd verl/examples/data_preprocess
python geo3k.py --local_save_dir ~/data/geo3k
```

## Quick Start

### 1. Test Data Loading (Optional)

```bash
python opentinker/test_geo3k_data.py --data_path ~/data/geo3k/train.parquet
```

### 2. Configure Training

Edit `opentinker/client/client_config/geo3k_param.yaml`:

```yaml
# Model paths
tokenizer_path: Qwen/Qwen2.5-VL-7B-Instruct
processor_path: Qwen/Qwen2.5-VL-7B-Instruct

# Data
data_path: ~/data/geo3k/train.parquet
val_data_path: ~/data/geo3k/test.parquet

# GRPO settings
adv_estimator: "grpo"
rollout_n: 5

# Resources
num_gpus: 8
batch_size: 64
```

### 3. Launch Training

```bash
python opentinker/client/geo3k_rl.py
```

Or with custom parameters:

```bash
python opentinker/client/geo3k_rl.py \
  tokenizer_path=Qwen/Qwen2.5-VL-7B-Instruct \
  batch_size=32 \
  num_epochs=15 \
  num_gpus=4
```

## Architecture Components

- **Data Generator**: `StaticDatasetGeneratorVL` - loads images from parquet
- **Dataset**: `DynamicGameDatasetVL` - processes text + images with AutoProcessor
- **Environment**: `VLGameEnvironment` - VL-aware training environment
- **Game**: `Geo3KGame` - geometry problem logic with reward computation
- **Client**: `geo3k_rl.py` - training launcher

## Key Differences from Text-Only Training

| Aspect | Text-Only | Vision-Language |
|--------|-----------|-----------------|
| Processor | AutoTokenizer | AutoProcessor |
| Data Generator | StaticDatasetGenerator | StaticDatasetGeneratorVL |
| Dataset | DynamicGameDataset | DynamicGameDatasetVL |
| Environment | GameEnvironment | VLGameEnvironment |
| Data Fields | prompt | prompt + images |
| Model Input | input_ids, attention_mask | + pixel_values, image_grid_thw |

## Next Steps

### Add Multi-Turn Support

Create a multi-turn version that allows reasoning refinement:

1. Extend `Geo3KGame` for multi-turn interactions
2. Update config: `max_user_turns: 2`, `max_assistant_turns: 3`
3. Optionally add tools for intermediate verification

### Add Other VL Tasks

Follow the Geo3K pattern for:
- MathVista (math with diagrams)
- ChartQA (chart understanding)
- DocVQA (document QA)

## Troubleshooting

### "No module named transformers"
```bash
pip install transformers>=4.37.0
```

### "Data file not found"
```bash
python verl/examples/data_preprocess/geo3k.py --local_save_dir ~/data/geo3k
```

### "AutoProcessor not found"
Ensure you're using a VL model path (e.g., Qwen2.5-VL, not Qwen2.5).

## References

- Implementation Plan: `implementation_plan.md`
- Walkthrough: `walkthrough.md`
- verl Geo3K Example: `verl/examples/grpo_trainer/run_qwen2_5_vl-7b.sh`
