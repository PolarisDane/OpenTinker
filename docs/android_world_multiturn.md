# LLM Game Agent (AndroidWorld Multi-Turn)

This example demonstrates training a language model to complete tasks in the Android operating system environment using AndroidWorld.

## Overview

**AndroidWorld** is a dynamic benchmarking environment for autonomous agents to interact with the Android operating system. The agent perceives the screen via a list of UI elements and interacts by performing actions like clicking, typing, and scrolling.

Tasks include:
- Adding contacts
- Managing settings
- Browsing information
- Sending messages
- And more...

## Prerequisites

1.  Complete the [Installation](../README.md#-installation) steps.
2.  **Environment Setup**: You must install the Android SDK and run an Emulator. See the **[Detailed Environment Setup](#detailed-environment-setup)** section below for instructions.
3.  Get your IP address: `hostname -I`

## Step 1: Start the Scheduler (Server Side)

```bash
bash opentinker/scripts/launch_scheduler.sh --scheduler-port <scheduler_port>
```

## Step 2: Start the AndroidWorld Environment (Server Side)

Before starting the environment server, ensure your Android Emulator is running (see setup below).

```bash
python -m opentinker.environment.android_world.android_world_server \
    --port 8092 \
    --max_steps 50 \
    --split train
```

**Server Options:**

- `--port`: Server port (default: 8082, recommend 8092 to match client config)
- `--max_steps`: Max steps per episode (default: 50)
- `--split`: Dataset split (`train`, `eval_in_distribution`, `eval_out_of_distribution`)
- `--shards`: Number of parallel server instances (for parallel training)

## Step 3: Run Training

```bash
python opentinker/client/android_world_rl.py \
    tokenizer_path=Qwen/Qwen2.5-3B-Instruct \
    batch_size=4 \
    val_batch_size=50 \
    num_steps=1000 \
    save_freq=20000 \
    test_freq=10 \
    scheduler_url=http://<server_endpoint>:<scheduler_port> \
    interaction.config.env_port=8092 \
    interaction.config.env_host=<env_server_endpoint>
```

**Training Parameters:**

- `num_steps`: Total training steps (alternative: use `num_epochs`)
- `batch_size`: Training batch size
- `val_batch_size`: Validation samples per evaluation
- `test_freq`: Validation frequency (every N steps)
- `adv_estimator`: Advantage estimator (`gae`, `grpo`, `grpo_per_step`)

## Reward Structure

| Event            | Reward |
| :--------------- | ------ |
| Task Success     | +10.0  |
| Task Failure     | -1.0   |
| Per Step Penalty | -0.01  |
| Invalid Action   | -0.1   |

## Example Actions

The agent interacts with the environment by outputting JSON commands referencing UI element indices:

- **Click**: `{"action_type": "click", "index": 4}`
- **Type**: `{"action_type": "input_text", "text": "Alice", "index": 2}`
- **Scroll**: `{"action_type": "scroll", "direction": "down"}`
- **Open App**: `{"action_type": "open_app", "app_name": "Settings"}`
- **Navigate Home**: `{"action_type": "navigate_home"}`
- **Navigate Back**: `{"action_type": "navigate_back"}`
- **Answer Question**: `{"action_type": "answer", "text": "It is 5 PM."}`
- **Finish Task**: `{"action_type": "status", "goal_status": "complete"}`

## Configuration Reference

See [`opentinker/client/client_config/android_world_param.yaml`](../opentinker/client/client_config/android_world_param.yaml) for full configuration options.

---

## Evaluation & Task Splits (ID / OOD)

The framework provides **independent train/test task splits** to properly measure in-distribution (ID) and out-of-distribution (OOD) generalization.

### Task Split Design

| Split | Purpose | Tasks |
|-------|---------|-------|
| **train** | RL training | 28 tasks: Contacts, Calendar, SMS, Markor (basic), System, Expense (basic), Files |
| **test_id** | In-Distribution eval | Same task *types* as train, fresh random parameters |
| **test_ood** | Out-of-Distribution eval | 60+ task types NOT seen during training (Recipe, Browser, Camera, Clock, OsmAnd, RetroMusic, VLC, composites, etc.) |

The config file is at [`opentinker/environment/android_world/task_sets.yaml`](../opentinker/environment/android_world/task_sets.yaml).

### Running Evaluation

**1. Validate config (no emulator needed):**
```bash
bash opentinker/scripts/run_android.sh eval-validate
```

**2. Run ID + OOD evaluation:**
```bash
# Requires running emulator
bash opentinker/scripts/run_android.sh eval
```

**3. Run evaluation with custom settings:**
```bash
EVAL_SPLIT="test_ood" EVAL_INSTANCES=5 EVAL_MAX_STEPS=30 \
    bash opentinker/scripts/run_android.sh eval
```

**4. Run eval script directly:**
```bash
python opentinker/environment/android_world/run_eval.py \
    --split test_id test_ood \
    --n_instances 3 \
    --max_steps 30 \
    --seed 42 \
    --output_dir ./eval_results
```

**5. Recompute metrics from saved results:**
```bash
python opentinker/environment/android_world/run_eval.py \
    --from_json eval_results/eval_test_ood_1708000000.json
```

### Evaluation Metrics

The evaluator reports:

| Metric | Description |
|--------|-------------|
| **Success Rate** | Fraction of episodes where the task was completed successfully |
| **Avg Steps (all)** | Average number of steps across all episodes |
| **Avg Steps (success)** | Average steps for successful episodes only |
| **Median Steps (success)** | Median steps for successful episodes |
| **Avg Reward** | Mean cumulative reward per episode |
| **Timeout Rate** | Fraction of episodes that hit the max step limit |
| **Avg Invalid Actions** | Mean number of invalid action penalties per episode |
| **Error Rate** | Fraction of episodes with runtime errors |
| **Per-Category Breakdown** | Metrics grouped by app category (Calendar, SMS, etc.) |
| **Per-Task Success Rate** | Success rate for each individual task type |

### Training with Task Set Config

The training env server now supports `--task_set_config` to ensure training only samples from the train split:

```bash
python opentinker/environment/android_world/android_world_server.py \
    --port 8092 \
    --shards 4 \
    --split train \
    --task_set_config opentinker/environment/android_world/task_sets.yaml
```

### Customizing Task Splits

Edit `opentinker/environment/android_world/task_sets.yaml`:

```yaml
train:
  tasks:
    - ContactsAddContact
    - SimpleSmsSend
    # ... add your training tasks

test_ood:
  tasks:
    - RecipeAddSingleRecipe
    - BrowserDraw
    # ... add your OOD test tasks

eval_settings:
  n_instances_per_task: 3   # instances per task type
  max_steps: 30             # max steps per episode
  seed: 42                  # reproducibility
```

### Programmatic Usage

```python
from opentinker.environment.android_world import TaskSetConfig, AndroidWorldEvaluator

# Load config
config = TaskSetConfig("opentinker/environment/android_world/task_sets.yaml")
print(config.summary())

# Get task lists
train_tasks = config.get_tasks("train")      # 28 tasks
test_id = config.get_tasks("test_id")        # same types, fresh params
test_ood = config.get_tasks("test_ood")      # 60+ novel tasks

# Sample for training
task = config.sample_task("train")

# Run evaluation
game = AndroidWorldGame(task_types=test_ood, max_steps=30)
evaluator = AndroidWorldEvaluator(game, config, split="test_ood", agent_fn=my_agent)
results = evaluator.run()
print(results.summary())
results.save("./eval_results")
```

---

## Detailed Environment Setup

### 1. Android SDK & Command Line Tools

If you do not have Android Studio installed, you can set up the command-line tools manually.

1.  **Create Directory Structure:**
    ```bash
    mkdir -p /usr/local/android-sdk/cmdline-tools
    cd /usr/local/android-sdk/cmdline-tools
    ```

2.  **Download Command Line Tools:**
    ```bash
    wget https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O cmdline-tools.zip
    unzip cmdline-tools.zip
    mv cmdline-tools latest
    rm cmdline-tools.zip
    ```

3.  **Install SDK Components:**
    ```bash
    export ANDROID_HOME=/usr/local/android-sdk
    export PATH=$ANDROID_HOME/cmdline-tools/latest/bin:$PATH

    # Accept licenses
    yes | sdkmanager --licenses --sdk_root=$ANDROID_HOME

    # Install Platform Tools (adb), Android 33 Platform, and Build Tools
    sdkmanager "platform-tools" "platforms;android-33" "build-tools;34.0.0" "emulator" --sdk_root=$ANDROID_HOME
    ```

4.  **Configure Environment Variables:**
    Add the following to your shell configuration file (`~/.bashrc` or `~/.zshrc`):
    ```bash
    export JAVA_HOME="/usr/local/android-studio/jbr" # Or your JDK path
    export ANDROID_HOME="/usr/local/android-sdk"
    export PATH="$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/emulator:$PATH"
    ```

### 2. Create Android Virtual Device (AVD)

Create an AVD named `AndroidWorldAvd` targeting Android 13 (Tiramisu, API 33).

1.  **Install System Image:**
    *   For x86_64 (Standard PC):
        ```bash
        sdkmanager "system-images;android-33;google_apis;x86_64" --sdk_root=$ANDROID_HOME
        ```
    *   For ARM64 (Apple Silicon or Software Emulation on x86):
        ```bash
        sdkmanager "system-images;android-33;google_apis;arm64-v8a" --sdk_root=$ANDROID_HOME
        ```

2.  **Create AVD:**
    ```bash
    echo "no" | avdmanager create avd --name AndroidWorldAvd --package "system-images;android-33;google_apis;x86_64" --device "pixel_6"
    ```
    *(Replace `x86_64` with `arm64-v8a` if applicable)*

### 3. Launch Emulator

Start the emulator in a separate terminal or background process using the `sg` command to ensure correct group permissions (e.g., `kvm`).

*   **Standard Launch (with GUI):**
    ```bash
    sg kvm -c "emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554"
    ```

*   **Headless Launch (Server/Docker):**
    ```bash
    sg kvm -c "emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 -no-window -no-audio"
    ```

*   **Software Emulation (No KVM):**
    If hardware acceleration is unavailable, add `-accel off`. **Warning: Performance will be very low.**
    ```bash
    emulator -avd AndroidWorldAvd -no-snapshot -grpc 8554 -no-window -no-audio -accel off
    ```

## Quick Start with `run_android.sh`

For multi-emulator parallel training, we provide an all-in-one launcher script [`opentinker/scripts/run_android.sh`](../opentinker/scripts/run_android.sh) that automates AVD creation, emulator startup, environment server, and training client.

### Usage

Run each step in a **separate terminal**:

```bash
# Step 0 (one-time): Create N AVDs for parallel training
bash opentinker/scripts/run_android.sh setup-avds

# Step 1: Start the scheduler
bash opentinker/scripts/run_android.sh scheduler

# Step 2: Start N Android emulators in parallel
bash opentinker/scripts/run_android.sh simulator

# Step 3: Start the sharded environment server (after emulators fully boot)
bash opentinker/scripts/run_android.sh env

# Step 4: Launch RL training
bash opentinker/scripts/run_android.sh client
```

### Environment Variables

All settings are configurable via environment variables:

| Variable | Default | Description |
| :------- | :------ | :---------- |
| `NUM_EMULATORS` | `4` | Number of parallel emulators |
| `NUM_GPUS` | `4` | Number of GPUs for model parallelism |
| `GPUS` | `[0,1,2,3]` | GPU device list |
| `MODEL_PATH` | `Qwen/Qwen2.5-3B-Instruct` | Model path or HuggingFace ID |
| `AVD_NAME` | `AndroidWorldAvd` | AVD name prefix (creates `{AVD_NAME}_0`, `{AVD_NAME}_1`, ...) |
| `EMULATOR_HEADLESS` | `1` | Set `0` to show emulator GUI |
| `EMULATOR_NO_KVM` | `0` | Set `1` for software emulation (slow) |
| `SCHEDULER_PORT` | `9780` | Scheduler listen port |
| `ENV_PORT` | `9092` | Environment server base port |

**Example** — scale to 8 emulators on 8 GPUs:

```bash
NUM_EMULATORS=8 NUM_GPUS=8 GPUS="[0,1,2,3,4,5,6,7]" bash opentinker/scripts/run_android.sh setup-avds
# Then run scheduler / simulator / env / client with the same env vars
```

---

## App Requirements per Task Split

The default environment setup (`setup_apps()` in `android_world/env/setup_device/setup.py`) installs **all 24 apps** automatically. The table below documents which apps each task split config file actually requires, useful for debugging or selective installation.

### Pre-installed Apps (no APK needed)

These ship with the Android 13 emulator image and require no installation:

| App | Package Name | Used By Tasks |
|-----|-------------|---------------|
| Camera | `com.android.camera2` | Camera\* |
| Chrome | `com.android.chrome` | Browser\* |
| Clock | `com.google.android.deskclock` | Clock\* |
| Contacts | `com.google.android.contacts` | Contacts\* |
| Dialer | `com.google.android.dialer` | (phone calls) |
| Files | `com.google.android.documentsui` | Files\* |
| Settings | `com.android.settings` | System\*, OpenApp\*, TurnOn/Off\* |

### Third-party Apps (APK installed during setup)

| App | Package Name | APK | Used By Tasks |
|-----|-------------|-----|---------------|
| Android World | `com.google.androidworld` | `androidworld.apk` | (infrastructure) |
| Audio Recorder | `com.dimowner.audiorecorder` | `com.dimowner.audiorecorder_926.apk` | AudioRecorder\* |
| Broccoli (Recipe) | `com.flauschcode.broccoli` | `com.flauschcode.broccoli_1020600.apk` | Recipe\* |
| Clipper | `ca.zgrs.clipper` | `clipper.apk` | (clipboard infrastructure) |
| Joplin | `net.cozic.joplin` | `net.cozic.joplin_2097740.apk` | Notes\* |
| Markor | `net.gsantner.markor` | `net.gsantner.markor_146.apk` | Markor\* |
| MiniWoB | — | `miniwobapp.apk` | (MiniWoB tasks) |
| OpenTracks | `de.dennisguse.opentracks` | `de.dennisguse.opentracks_5705.apk` | SportsTracker\* |
| OsmAnd | `net.osmand` | `net.osmand-4.6.13.apk` | OsmAnd\* |
| Pro Expense | `com.arduia.expense` | `com.arduia.expense_11.apk` | Expense\* |
| Retro Music | `code.name.monkey.retromusic` | `code.name.monkey.retromusic_10603.apk` | Retro\* |
| Simple Calendar Pro | `com.simplemobiletools.calendar.pro` | `com.simplemobiletools.calendar.pro_238.apk` | SimpleCalendar\* |
| Simple Draw Pro | `com.simplemobiletools.draw.pro` | `com.simplemobiletools.draw.pro_79.apk` | SimpleDrawPro\* |
| Simple Gallery Pro | `com.simplemobiletools.gallery.pro` | `com.simplemobiletools.gallery.pro_396.apk` | SaveCopyOfReceipt\*, Expense\*FromGallery |
| Simple SMS Messenger | `com.simplemobiletools.smsmessenger` | `com.simplemobiletools.smsmessenger_85.apk` | SimpleSms\*, \*AndSms composites |
| Tasks | `org.tasks` | `org.tasks_130605.apk` | Tasks\* |
| VLC | `org.videolan.vlc` | `org.videolan.vlc_13050408.apk` | Vlc\* |

### Per-Split App Requirements

| Split Config | Train Apps (APK) | OOD Apps (APK) | Pre-installed |
|-------------|------------------|----------------|---------------|
| `task_sets_split1_markor.yaml` | Markor | Markor, Simple SMS, Clipper | — |
| `task_sets_split2_expense_recipe.yaml` | Pro Expense, Markor, Simple Gallery Pro | Broccoli (Recipe), Markor | — |
| `task_sets_split3_system.yaml` | — | — | Settings |
| `task_sets_split4_calendar.yaml` | Simple Calendar Pro | Simple Calendar Pro | — |
| `task_sets_split5_sms_contacts.yaml` | Simple SMS | Audio Recorder, OsmAnd, Retro Music, Simple Draw Pro, VLC | Contacts, Chrome, Camera, Clock |
| `task_sets_split6_calendar_inapp.yaml` | Simple Calendar Pro | Simple Calendar Pro | — |
| `task_sets_split7_expense_difficulty.yaml` | Pro Expense | Pro Expense, Simple Gallery Pro, Markor | — |
| `task_sets_split8_expense_operation.yaml` | Pro Expense, Markor, Simple Gallery Pro | Pro Expense | — |
| `task_sets_split9_markor_mixed.yaml` | Markor, Clipper | Markor, Simple SMS | — |
| `task_sets_split10_calendar_mixed.yaml` | Simple Calendar Pro | Simple Calendar Pro | — |
| `task_sets_split11_expense_mixed.yaml` | Pro Expense, Markor, Simple Gallery Pro | Pro Expense | — |

> **Note:** All apps listed above are included in the default `_APPS` installation list. No additional installation steps are required beyond the standard `setup_apps()` call during environment initialization.

---

## Troubleshooting

*   **"KVM is not found"**: Ensure virtualization is enabled in your BIOS/Hypervisor. On Linux, check permissions for `/dev/kvm`. If in a container, run with `--device /dev/kvm`.
*   **Emulator crashes immediately**: Check logs. If running x86_64 image on ARM or vice-versa, the emulator will fail. Use the correct system image for your host architecture.
*   **"ADB command not found"**: Ensure `platform-tools` is in your `$PATH`.
*   **"Process system isn't responding"**: Common in software emulation (`-accel off`). Wait for the system to stabilize or dismiss the dialog.