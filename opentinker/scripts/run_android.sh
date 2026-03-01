#!/bin/bash
# AndroidWorld Training Script (Multi-Turn, Multi-Emulator)
#
# This script runs AndroidWorld RL training with OpenTinker.
# You need to run these steps in SEPARATE terminals.
#
# For Training (4 terminals):
#   Terminal 1: bash run_android.sh scheduler
#   Terminal 2: bash run_android.sh simulator    # Android Emulator (start BEFORE env)
#   Terminal 3: bash run_android.sh env
#   Terminal 4: bash run_android.sh client
#
# Prerequisites:
#   - Android SDK, AVD "AndroidWorldAvd" (or set AVD_NAME), and emulator in PATH
#   - See docs/android_world_multiturn.md for environment setup

# =============================================================================
# Configuration
# =============================================================================
SCHEDULER_PORT=9780
ENV_PORT=9092
GPUS="${GPUS:-[0,1,2,3]}"
NUM_GPUS="${NUM_GPUS:-4}"  # For tensor_model_parallel_size (model spans N GPUs)

# Multi-emulator configuration
# Set NUM_EMULATORS to match NUM_GPUS for true parallelism
NUM_EMULATORS="${NUM_EMULATORS:-4}"

# Emulator (simulator) base ports
AVD_NAME="${AVD_NAME:-AndroidWorldAvd}"
# Console ports: 5556, 5558, 5560, 5562 (each +2 because ADB uses console+1)
EMULATOR_BASE_CONSOLE_PORT="${EMULATOR_BASE_CONSOLE_PORT:-5556}"
# gRPC ports: 8554, 8555, 8556, 8557
EMULATOR_BASE_GRPC_PORT="${EMULATOR_BASE_GRPC_PORT:-8554}"
# EMULATOR_HEADLESS=1  -> -no-window -no-audio
EMULATOR_HEADLESS="${EMULATOR_HEADLESS:-1}"
# EMULATOR_NO_KVM=1    -> no "sg kvm", add -accel off (slow, for hosts without KVM)
EMULATOR_NO_KVM="${EMULATOR_NO_KVM:-0}"

# Fix vLLM v1 cumem allocator issue
export VLLM_DISABLE_SLEEP_MODE=1

# Model path (set to your model path)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"

# OpenTinker root (relative to this script: opentinker/scripts/run_android.sh)
OPENTINKER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Activate conda environment (adjust to your setup)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
# conda activate <your_env_name>

# Change to OpenTinker directory
cd "$OPENTINKER_ROOT"

# =============================================================================
# Step Selection
# =============================================================================
case "$1" in
    setup-avds)
        echo "========================================"
        echo "Creating $NUM_EMULATORS AVDs for parallel training"
        echo "========================================"
        echo ""
        echo "This will create AVDs named: ${AVD_NAME}_0, ${AVD_NAME}_1, ..., ${AVD_NAME}_$((NUM_EMULATORS-1))"
        echo ""
        
        # Detect system image (x86_64 or arm64-v8a)
        SYSTEM_IMAGE="${SYSTEM_IMAGE:-system-images;android-33;google_apis;x86_64}"
        echo "Using system image: $SYSTEM_IMAGE"
        echo ""
        
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            AVD_NAME_I="${AVD_NAME}_${i}"
            echo "Creating AVD: $AVD_NAME_I"
            
            # Check if AVD already exists
            if avdmanager list avd | grep -q "Name: $AVD_NAME_I"; then
                echo "  AVD $AVD_NAME_I already exists, skipping..."
            else
                echo "no" | avdmanager create avd \
                    --name "$AVD_NAME_I" \
                    --package "$SYSTEM_IMAGE" \
                    --device "pixel_6" \
                    --force
                echo "  Created $AVD_NAME_I"
            fi
        done
        
        echo ""
        echo "Done! Created $NUM_EMULATORS AVDs."
        echo "You can now run: bash run_android.sh simulator"
        ;;

    scheduler|1)
        echo "========================================"
        echo "Step 1: Starting Scheduler on port $SCHEDULER_PORT"
        echo "========================================"
        bash opentinker/scripts/launch_scheduler.sh \
            --scheduler-port $SCHEDULER_PORT \
            --gpus "$GPUS"
        ;;

    simulator|2)
        echo "========================================"
        echo "Step 2: Starting $NUM_EMULATORS Android Emulators"
        echo "  AVD base name: ${AVD_NAME}_0 ... ${AVD_NAME}_$((NUM_EMULATORS-1))"
        echo "  Base Console Port=$EMULATOR_BASE_CONSOLE_PORT"
        echo "  Base gRPC Port=$EMULATOR_BASE_GRPC_PORT"
        echo "========================================"
        echo ""
        echo "IMPORTANT: Before starting, ensure NO other emulators are running!"
        echo "  Check: adb devices"
        echo "  Kill all emulators: adb emu kill (or close emulator windows)"
        echo ""
        echo "Starting $NUM_EMULATORS emulators:"
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            CONSOLE_PORT=$((EMULATOR_BASE_CONSOLE_PORT + i * 2))
            GRPC_PORT=$((EMULATOR_BASE_GRPC_PORT + i))
            echo "  Emulator $i: console=$CONSOLE_PORT (ADB=$((CONSOLE_PORT + 1))), grpc=$GRPC_PORT"
        done
        echo ""
        echo "Ensure the env server is started AFTER all emulators are fully booted."
        echo ""

        # Check if AVDs exist
        echo "Checking AVDs..."
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            AVD_NAME_I="${AVD_NAME}_${i}"
            if ! avdmanager list avd 2>/dev/null | grep -q "Name: $AVD_NAME_I"; then
                echo "ERROR: AVD '$AVD_NAME_I' not found!"
                echo "Run 'bash run_android.sh setup-avds' first to create the AVDs."
                exit 1
            fi
        done
        echo "All AVDs found."
        echo ""

        # Start all emulators in background
        PIDS=()
        for i in $(seq 0 $((NUM_EMULATORS - 1))); do
            AVD_NAME_I="${AVD_NAME}_${i}"
            CONSOLE_PORT=$((EMULATOR_BASE_CONSOLE_PORT + i * 2))
            GRPC_PORT=$((EMULATOR_BASE_GRPC_PORT + i))
            
            BASE="emulator -avd $AVD_NAME_I -no-snapshot -port $CONSOLE_PORT -grpc $GRPC_PORT"
            if [ "$EMULATOR_NO_KVM" = "1" ]; then
                CMD="$BASE -no-window -no-audio -accel off"
            elif [ "$EMULATOR_HEADLESS" = "1" ]; then
                CMD="$BASE -no-window -no-audio"
            else
                CMD="$BASE"
            fi
            
            echo "Starting emulator $i ($AVD_NAME_I): $CMD"
            if [ "$EMULATOR_NO_KVM" = "1" ]; then
                $CMD &
            else
                sg kvm -c "$CMD" &
            fi
            PIDS+=($!)
            sleep 2  # Wait a bit between emulator starts
        done
        
        echo ""
        echo "All $NUM_EMULATORS emulators started. PIDs: ${PIDS[*]}"
        echo "Waiting for all emulators... Press Ctrl+C to stop."
        wait
        ;;

    env|3)
        echo "========================================"
        echo "Step 3: Starting AndroidWorld Environment Server"
        echo "  Shards: $NUM_EMULATORS (ports $ENV_PORT..$((ENV_PORT + NUM_EMULATORS - 1)))"
        echo "  Emulator base ports: console=$EMULATOR_BASE_CONSOLE_PORT, grpc=$EMULATOR_BASE_GRPC_PORT"
        echo "========================================"
        echo "Make sure all $NUM_EMULATORS Android Emulators are running first."
        echo ""
        python opentinker/environment/android_world/android_world_server.py \
            --port $ENV_PORT \
            --shards $NUM_EMULATORS \
            --emulator_base_console_port $EMULATOR_BASE_CONSOLE_PORT \
            --emulator_base_grpc_port $EMULATOR_BASE_GRPC_PORT \
            --split train \
            --max_steps 50 \
            --task_set_config opentinker/environment/android_world/task_sets.yaml
        ;;

    client|4)
        echo "========================================"
        echo "Step 4: Running AndroidWorld RL Client"
        echo "  Emulators: $NUM_EMULATORS (parallel rollouts)"
        echo "  GPUs: $NUM_GPUS (tensor parallelism)"
        echo "========================================"
        # Multi-emulator parallel training:
        # - batch_size=NUM_GPUS: satisfies batch_size >= num_gpus for data partitioning
        # - agent_num_workers=NUM_EMULATORS: parallel rollouts (one per emulator)
        # - env_shards=NUM_EMULATORS: routes requests to different env servers/emulators
        # - num_gpus=NUM_GPUS: model tensor parallelism (solves OOM)
        python opentinker/client/android_world_rl.py \
            tokenizer_path=$MODEL_PATH \
            batch_size=$NUM_GPUS \
            val_batch_size=$NUM_GPUS \
            rollout_n=1 \
            adv_estimator=gae \
            agent_num_workers=$NUM_EMULATORS \
            num_steps=1000 \
            save_freq=50 \
            test_freq=10 \
            num_gpus=$NUM_GPUS \
            scheduler_url=http://0.0.0.0:$SCHEDULER_PORT \
            interaction.config.env_port=$ENV_PORT \
            interaction.config.env_host=0.0.0.0 \
            interaction.config.env_shards=$NUM_EMULATORS
        ;;

    eval)
        # =====================================================================
        # Evaluation: runs ID and OOD test sets
        # =====================================================================
        EVAL_SPLIT="${EVAL_SPLIT:-test_id test_ood}"
        EVAL_INSTANCES="${EVAL_INSTANCES:-3}"
        EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-30}"
        EVAL_SEED="${EVAL_SEED:-42}"
        EVAL_OUTPUT="${EVAL_OUTPUT:-./eval_results}"
        EVAL_NUM_EMULATORS="${EVAL_NUM_EMULATORS:-1}"
        # Checkpoint / model to evaluate (optional)
        EVAL_MODEL_PATH="${EVAL_MODEL_PATH:-}"
        EVAL_TOKENIZER_PATH="${EVAL_TOKENIZER_PATH:-}"
        EVAL_VLLM_SERVER_URL="${EVAL_VLLM_SERVER_URL:-}"
        EVAL_TENSOR_PARALLEL_SIZE="${EVAL_TENSOR_PARALLEL_SIZE:-1}"
        EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.9}"
        EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
        EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-4096}"
        echo "========================================"
        echo "Running Android World Evaluation"
        echo "  Splits: $EVAL_SPLIT"
        echo "  Instances per task: $EVAL_INSTANCES"
        echo "  Max steps: $EVAL_MAX_STEPS"
        echo "  Seed: $EVAL_SEED"
        echo "  Output: $EVAL_OUTPUT"
        echo "  Emulators: $EVAL_NUM_EMULATORS"
        if [ -n "$EVAL_MODEL_PATH" ]; then
            echo "  Model: $EVAL_MODEL_PATH"
        elif [ -n "$EVAL_VLLM_SERVER_URL" ]; then
            echo "  vLLM Server: $EVAL_VLLM_SERVER_URL"
        else
            echo "  Model: (none — dummy agent)"
        fi
        echo "========================================"

        # Build optional model arguments
        MODEL_ARGS=""
        if [ -n "$EVAL_MODEL_PATH" ]; then
            MODEL_ARGS="$MODEL_ARGS --model_path $EVAL_MODEL_PATH"
        fi
        if [ -n "$EVAL_TOKENIZER_PATH" ]; then
            MODEL_ARGS="$MODEL_ARGS --tokenizer_path $EVAL_TOKENIZER_PATH"
        fi
        if [ -n "$EVAL_VLLM_SERVER_URL" ]; then
            MODEL_ARGS="$MODEL_ARGS --vllm_server_url $EVAL_VLLM_SERVER_URL"
        fi
        MODEL_ARGS="$MODEL_ARGS --tensor_parallel_size $EVAL_TENSOR_PARALLEL_SIZE"
        MODEL_ARGS="$MODEL_ARGS --gpu_memory_utilization $EVAL_GPU_MEM_UTIL"
        MODEL_ARGS="$MODEL_ARGS --temperature $EVAL_TEMPERATURE"
        MODEL_ARGS="$MODEL_ARGS --max_tokens $EVAL_MAX_TOKENS"

        python opentinker/environment/android_world/run_eval.py \
            --split $EVAL_SPLIT \
            --n_instances $EVAL_INSTANCES \
            --max_steps $EVAL_MAX_STEPS \
            --seed $EVAL_SEED \
            --output_dir $EVAL_OUTPUT \
            --num_emulators $EVAL_NUM_EMULATORS \
            --emulator_base_console_port $EMULATOR_BASE_CONSOLE_PORT \
            --emulator_base_grpc_port $EMULATOR_BASE_GRPC_PORT \
            $MODEL_ARGS
        ;;

    eval-validate)
        # =====================================================================
        # Validate task_sets.yaml config (no emulator needed)
        # =====================================================================
        echo "========================================"
        echo "Validating Android World Task Set Config"
        echo "========================================"
        python opentinker/environment/android_world/run_eval.py --validate_only
        ;;

    *)
        echo "AndroidWorld Training Script (Multi-Turn, Multi-Emulator)"
        echo ""
        echo "Usage: $0 {setup-avds|scheduler|simulator|env|client|eval|eval-validate}"
        echo "       $0 {1|2|3|4}"
        echo ""
        echo "=== First Time Setup ==="
        echo "  $0 setup-avds            # Create $NUM_EMULATORS AVDs for parallel training"
        echo ""
        echo "=== For Training (4 terminals) ==="
        echo "  Terminal 1: $0 scheduler   # Start scheduler (port $SCHEDULER_PORT)"
        echo "  Terminal 2: $0 simulator   # Start $NUM_EMULATORS Android Emulators (start BEFORE env)"
        echo "  Terminal 3: $0 env         # Start $NUM_EMULATORS env server shards (ports $ENV_PORT..$((ENV_PORT+NUM_EMULATORS-1)))"
        echo "  Terminal 4: $0 client      # Start RL training client"
        echo ""
        echo "=== For Evaluation ==="
        echo "  $0 eval                    # Run ID + OOD evaluation (needs running emulator)"
        echo "  EVAL_NUM_EMULATORS=4 $0 eval  # Parallel eval on 4 emulators"
        echo "  $0 eval-validate           # Validate task_sets.yaml (no emulator needed)"
        echo ""
        echo "Evaluation Configuration (env vars):"
        echo "  EVAL_MODEL_PATH=<path>     # Checkpoint or model to evaluate (e.g., ckpt/step_100/)"
        echo "  EVAL_TOKENIZER_PATH=<path> # Tokenizer path (defaults to EVAL_MODEL_PATH)"
        echo "  EVAL_VLLM_SERVER_URL=<url> # vLLM server URL (alternative to local model)"
        echo "  EVAL_TENSOR_PARALLEL_SIZE=1  # Tensor parallelism for local model"
        echo "  EVAL_GPU_MEM_UTIL=0.9      # GPU memory fraction"
        echo "  EVAL_TEMPERATURE=0.0       # Sampling temperature (0=greedy)"
        echo "  EVAL_MAX_TOKENS=4096       # Max tokens per action"
        echo "  EVAL_NUM_EMULATORS=1       # Number of emulators for parallel eval"
        echo "  EVAL_SPLIT='test_id test_ood'  # Splits to evaluate"
        echo "  EVAL_INSTANCES=3           # Instances per task"
        echo "  EVAL_MAX_STEPS=30          # Max steps per episode"
        echo "  EVAL_SEED=42               # Random seed"
        echo ""
        echo "Emulator Configuration (env vars):"
        echo "  NUM_EMULATORS=$NUM_EMULATORS        # Number of parallel emulators"
        echo "  AVD_NAME=$AVD_NAME            # AVD base name (creates ${AVD_NAME}_0, ${AVD_NAME}_1, ...)"
        echo "  EMULATOR_BASE_CONSOLE_PORT=$EMULATOR_BASE_CONSOLE_PORT   # Base console port"
        echo "  EMULATOR_BASE_GRPC_PORT=$EMULATOR_BASE_GRPC_PORT      # Base gRPC port"
        echo "  EMULATOR_HEADLESS=1       # Headless: -no-window -no-audio"
        echo "  EMULATOR_NO_KVM=1         # No KVM: -accel off (slow, for containers/no KVM)"
        echo ""
        echo "IMPORTANT: Before running, ensure no other emulators are running!"
        echo "  Check: adb devices"
        echo "  Kill all: adb emu kill"
        echo ""
        echo "Configuration:"
        echo "  SCHEDULER_PORT=$SCHEDULER_PORT"
        echo "  ENV_PORT=$ENV_PORT"
        echo "  NUM_EMULATORS=$NUM_EMULATORS"
        echo "  GPUS=$GPUS"
        echo "  NUM_GPUS=$NUM_GPUS"
        echo "  MODEL_PATH=$MODEL_PATH"
        echo ""
        echo "See docs/android_world_multiturn.md for Android SDK and AVD setup."
        ;;
esac
