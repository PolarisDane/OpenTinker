# 优雅关闭（Graceful Shutdown）实现说明

## 概述

已实现优雅关闭机制，确保在进程被 kill 时正确清理资源。

---

## 1. 客户端优雅关闭

**文件**: `client/custom_client_with_scheduler.py`

### 功能

当客户端被中断时（Ctrl+C 或 `kill`）：
1. ✅ **取消调度器中的任务** - 调用 `scheduler.cancel_job(job_id)`
2. ✅ **清理 Ray actors** - 调度器会清理该任务关联的所有 Ray actors
3. ✅ **释放 GPU 资源** - GPU 和端口资源返回池中
4. ✅ **关闭奖励服务器** - 如果启动了奖励服务器，会被正确关闭

### 实现细节

```python
# 信号处理器
def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM for graceful shutdown"""
    signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
    print(f"\n⚠️  Received {signal_name} - Initiating graceful shutdown")
    
    # 清理任务
    cleanup_job()           # 取消调度器中的任务
    cleanup_reward_server() # 关闭奖励服务器
    
    sys.exit(0)

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command
```

### 支持的信号

- **SIGINT** (Ctrl+C) - 交互式中断
- **SIGTERM** (`kill <pid>`) - 终止信号

### 清理顺序

1. 从调度器取消任务
2. 调度器清理 Ray actors（`job_{job_id}_*`）
3. 调度器释放 GPU 和端口资源
4. 关闭本地奖励服务器进程
5. 优雅退出

---

## 2. 调度器优雅关闭

**文件**: `scheduler/launch_scheduler.py`

### 功能

当调度器被中断时（Ctrl+C 或 `kill`）：
1. ✅ **终止调度器 Actor** - `ray.kill(scheduler_actor)`
2. ✅ **关闭全局 Ray** - `ray.shutdown()`
3. ✅ **清理所有运行中的任务** - 自动清理所有子进程和 Ray actors

### 实现细节

```python
# 信号处理器
def signal_handler(signum, frame):
    """Handle SIGINT and SIGTERM for graceful shutdown"""
    signal_name = 'SIGINT' if signum == signal.SIGINT else 'SIGTERM'
    logger.info(f"⚠️ Received {signal_name} - Initiating graceful shutdown")
    
    cleanup_scheduler()  # 清理所有资源
    sys.exit(0)

def cleanup_scheduler():
    """Clean up scheduler resources on shutdown"""
    # 1. 终止调度器 actor
    if scheduler_actor_instance:
        ray.kill(scheduler_actor_instance)
    
    # 2. 关闭 Ray
    if ray.is_initialized():
        ray.shutdown()

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
```

### Ray 清理说明

调用 `ray.shutdown()` 会：
- 停止所有 Ray actors
- 清理所有 Ray 进程
- 释放所有 Ray 相关资源
- 关闭 Ray 集群连接

---

## 使用示例

### 场景 1: 客户端意外中断

```bash
# 启动客户端
python client/custom_client_with_scheduler.py \
    data_path=data/math/train.parquet \
    num_epochs=10

# 按 Ctrl+C 或在另一个终端 kill
kill <client_pid>

# 输出:
# ⚠️  Received SIGINT - Initiating graceful shutdown
# ====================================================
# 🧹 Cleaning up job abc123 from scheduler...
# ✓ Job abc123 cancelled and resources released
# 
# Shutting down reward server...
# ✓ Reward server stopped
# 
# 👋 Shutdown complete. Exiting...
```

**结果**:
- ✅ 任务从调度器移除
- ✅ GPU 资源释放
- ✅ 下一个排队的任务自动开始
- ✅ 没有僵尸进程

### 场景 2: 调度器关闭

```bash
# 启动调度器
python scheduler/launch_scheduler.py available_gpus=[0,1,2,3]

# 按 Ctrl+C
^C

# 输出:
# ⚠️ Received SIGINT - Initiating graceful shutdown
# ====================================================
# 🧹 Cleaning up scheduler resources...
# ====================================================
# Shutting down scheduler actor...
# ✓ Scheduler actor terminated
# Shutting down Ray...
# ✓ Ray shutdown complete
# ====================================================
# 👋 Scheduler cleanup complete
# ====================================================
```

**结果**:
- ✅ 所有运行中的任务被终止
- ✅ 所有 Ray actors 被清理
- ✅ Ray 集群正确关闭
- ✅ 没有残留进程

---

## 验证清理效果

### 检查 Ray 进程

```bash
# 关闭前
ps aux | grep ray
# 应该看到很多 ray:: 进程

# 关闭后
ps aux | grep ray
# 应该没有 ray:: 进程
```

### 检查 GPU 占用

```bash
# 关闭前
nvidia-smi
# GPU 有占用

# 关闭后
nvidia-smi
# GPU 内存释放
```

### 检查端口占用

```bash
# 检查调度器端口
lsof -i :8765

# 检查服务器端口
lsof -i :38001
```

---

## 注意事项

### 1. SIGKILL 无法捕获

`kill -9 <pid>` (SIGKILL) 无法被捕获，会导致资源未清理。

**建议**: 优先使用：
- `kill <pid>` (SIGTERM) - 可被捕获
- Ctrl+C (SIGINT) - 可被捕获

### 2. 客户端需要网络连接

客户端的清理依赖于能连接到调度器API。如果网络断开，清理会失败但不会阻塞退出。

### 3. 调度器清理所有任务

调度器关闭会终止**所有**运行中的任务，不仅是当前用户的任务。

### 4双重保险

代码中使用了双重保险机制：
- `signal` 模块捕获信号
- `atexit` 模块在正常退出时也会清理
- `try-finally` 确保异常时也清理

---

## 故障排查

### 问题: 客户端退出但任务仍在运行

**可能原因**:
- 使用了 `kill -9`
- 网络断开无法访问调度器

**解决方案**:
```bash
# 手动取消任务
curl -X DELETE http://localhost:8765/cancel_job/<job_id> \
    -H "Authorization: Bearer YOUR_API_KEY"
```

### 问题: Ray 进程没有清理

**可能原因**:
- 调度器使用了 `kill -9`
- Ray 初始化在其他地方

**解决方案**:
```bash
# 手动关闭 Ray
ray stop
# 或强制杀死所有 Ray 进程
pkill -9 -f ray::
```

---

## 总结

| 组件 | 信号 | 清理内容 | 文件 |
|------|------|---------|-----|
| **客户端** | SIGINT/SIGTERM | 取消任务、关闭奖励服务器 | `client/custom_client_with_scheduler.py` |
| **调度器** | SIGINT/SIGTERM | 终止 actor、关闭 Ray | `scheduler/launch_scheduler.py` |

**关键优势**:
- ✅ 资源自动清理
- ✅ 无僵尸进程
- ✅ GPU 正确释放
- ✅ 用户友好的日志输出
