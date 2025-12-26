# Scheduler & Web Dashboard Guide

This guide covers configuration and usage for the OpenTinker Job Scheduler and Web Dashboard.

## Configuration

The scheduler is configured via `opentinker/scheduler/config/scheduler.yaml`.

### Key Settings

```yaml
# Authentication
enable_auth: true # Set to true to require API keys
user_db_path: "scheduler_users.db"

# Resources
available_gpus: [0, 1, 2, 3] # GPUs to manage
port_range: null # null for auto-detect, or [min, max]
num_ports: 50 # Number of ports to auto-detect
scheduler_port: 8765 # Main API port
```

## Authentication

### 1. Registering Users

**Method 1: Interactive Script (Recommended)**

```bash
python opentinker/scheduler/register_user_example.py
```

This script prompts for a username, registers the user, and saves the API key to a local file.

**Method 2: REST API**

```bash
# Register a new user
curl -X POST "http://<scheduler_url>/register?username=<your_username>"
```

**Response:**

```json
{
  "user_id": "user_abc123",
  "username": "your_username",
  "api_key": "otk_98b8db24ccd64c92e1fdd9a232e209fa",
  "message": "User registered successfully..."
}
```

> ⚠️ **Important**: Save your API key immediately! It cannot be retrieved after registration.

### 2. Using the API Key

Include the API key in the `Authorization` header for all requests:

**cURL**:

```bash
curl -H "Authorization: Bearer <your_api_key>" http://<scheduler_url>/list_jobs
```

**Python**:

```python
import requests
headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get(f"{scheduler_url}/list_jobs", headers=headers)
```

The Web Dashboard provides a real-time view of job status and resource usage.

### 1. Start the Dashboard

```bash
python opentinker/scheduler/web_dashboard.py --port 8081
```

### 2. Access

Open [http://localhost:8081/web_dashboard.html](http://localhost:8081/web_dashboard.html) in your browser.

### 3. Authentication

If `enable_auth` is true in the scheduler config, you must provide an API Key.

1.  **Get your API Key**:
    - Run: `python opentinker/scheduler/register_user_example.py`
    - Or check your client config: `cat client/client_config/opentinker_param.yaml | grep scheduler_api_key`
2.  **Enter in Dashboard**:
    - Go to the "Settings" section at the top of the dashboard.
    - Paste your key into the "API Key" field.
    - The key is automatically saved to your browser's local storage.

## Scheduler API Reference

Base URL: `http://localhost:<scheduler_port>`

| Method | Endpoint                 | Description                                |
| ------ | ------------------------ | ------------------------------------------ |
| POST   | `/submit_job`            | Submit a new training job                  |
| GET    | `/list_jobs`             | List all jobs and their status             |
| GET    | `/job_status/{job_id}`   | Get details for a specific job             |
| DELETE | `/cancel_job/{job_id}`   | Cancel a running or queued job             |
| POST   | `/complete_job/{job_id}` | Mark a job as completed (called by client) |
| POST   | `/register`              | Register a new user (if auth enabled)      |

## Troubleshooting

### Job stuck in QUEUED

- Check GPU availability with `nvidia-smi`.
- Verify the scheduler has free ports in its range.

### 401 Unauthorized Errors

- Ensure you are providing a valid `Authorization: Bearer <key>` header (API) or have entered the key in the dashboard.
- If running locally without need for auth, set `enable_auth: false` in `scheduler.yaml`.

### Server Launch Failures

- Check the scheduler console logs for Python tracebacks.
- Ensure all dependencies are installed in the environment where the scheduler runs.
