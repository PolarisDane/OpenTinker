# 服务器连接超时问题 - 解决方案

## 问题描述

客户端在尝试连接到 HTTP 训练服务器时失败，报错：
```
RuntimeError: Failed to complete request to set_generation_config after 3 attempts
```

## 根本原因

**服务器初始化时间过长**：HTTP 训练服务器需要时间来：
1. 启动 Ray actors
2. 加载大型语言模型到 GPU
3. 初始化各个组件（actor, critic, reference model 等）

而客户端的重试机制太"急躁"：
- 只重试 3 次
- 重试间隔较短（2s、4s、8s）
- 总共只等待约 14 秒

对于需要加载大模型的服务器，这个时间远远不够！

---

## 已实施的解决方案

### 1. ✅ 增加 HTTP 客户端重试次数和延迟

**修改文件**: `client/http_training_client.py`

**更改内容**:
- `max_retries`: 3 → **10** (增加重试次数)
- `retry_delay`: 2.0s → **5.0s** (增加基础延迟)
- 添加指数退避上限：最长等待 60 秒

**新的重试时间表**:
```
尝试 1: 0秒
尝试 2: 5秒 →  等待 5s
尝试 3: 10秒 → 等待 5s  
尝试 4: 20秒 → 等待 10s
尝试 5: 40秒 → 等待 20s
尝试 6: 80秒 → 等待 40s
尝试 7: 140秒 → 等待 60s (上限)
尝试 8: 200秒 → 等待 60s
尝试 9: 260秒 → 等待 60s
尝试 10: 320秒 → 等待 60s
```

**总等待时间**: ~5 分钟（足够服务器完全启动）

### 2. ✅ 创建服务器就绪等待工具

**新文件**: `scripts/wait_for_server.py`

这是一个独立工具，可以在客户端连接之前检查服务器是否已就绪。

**用法**:
```bash
# 等待服务器就绪（默认超时 5 分钟）
python scripts/wait_for_server.py http://localhost:38001

# 自定义超时和轮询间隔
python scripts/wait_for_server.py http://localhost:38001 600 10
```

**输出示例**:
```
⏳ 等待服务器就绪: http://localhost:38001
   超时时间: 300秒
   检查端点: /api/v1/health

⏳ 尝试 1: 连接失败 (已等待: 0秒)
⏳ 尝试 2: HTTP 404 (已等待: 5秒)
⏳ 尝试 3: HTTP 200 ✓

✅ 服务器已就绪！(耗时: 47.2秒)
```

---

## 如何使用

### 方法 1: 直接运行（推荐）

由于已经修改了 `http_training_client.py`，现在客户端会自动等待更长时间：

```bash
python client/custom_client_with_scheduler.py \
    data_path=data/math/train.parquet \
    val_data_path=data/math/test.parquet \
    tokenizer_path=/path/to/tokenizer \
    num_gpus=4
```

客户端会自动：
- 重试 10 次（而不是 3 次）
- 使用指数退避策略
- 最多等待约 5 分钟

### 方法 2: 手动等待（最保险）

如果服务器特别慢，可以先手动等待：

```bash
# 步骤 1: 提交任务到调度器
# （从客户端输出中获取 server_url，比如 http://localhost:38001）

# 步骤 2: 等待服务器就绪
python scripts/wait_for_server.py http://localhost:38001 600

# 步骤 3: 服务器就绪后，客户端会自动连接
```

### 方法 3: 增加调度器的服务器启动等待时间

**修改文件**: `scheduler/job_scheduler.py`

找到第 394 行：
```python
time.sleep(0.5)  # Current: 0.5 seconds
```

改为：
```python
time.sleep(10.0)  # Wait 10 seconds for server to initialize
```

这会让调度器在启动服务器后等待更长时间再返回给客户端。

---

## 故障排查

### 检查服务器是否真的在运行

```bash
ps aux | grep launch_http_server
```

应该看到类似：
```
root  3283256  8.6  0.0  python .../launch_http_server.py server.port=38001 ...
```

### 检查服务器端口是否可访问

```bash
curl http://localhost:38001/api/v1/health
```

**预期响应**:
- 如果服务器还在初始化：`{"detail":"Not Found"}` 或连接错误
- 如果服务器已就绪：`{"status": "healthy", ...}`

### 查看服务器日志

服务器日志保存在 `/workspace/logs/`:
```bash
# 查找最新的日志文件
ls -lth /workspace/logs/ | head -5

# 查看stderr（错误日志）
tail -100 /workspace/logs/job_*_stderr.log
```

### 延长客户端超时时间

如果服务器特别慢（例如首次加载超大模型），可以在客户端创建时增加超时：

```python
# 在 custom_client_with_scheduler.py 中
client = ServiceClient(
    server_url=server_url,
    timeout=10000.0,  # 增加到 10000 秒
    max_retries=15,   # 增加到 15 次
    retry_delay=10.0  # 增加基础延迟到 10 秒
)
```

---

## 预防措施

### 1. 预热服务器

在第一次提交任务前，先启动一个服务器让它加载模型：

```bash
# 启动一个"预热"服务器
python server/launch_http_server.py server.port=38000

# 等待模型加载完成（观察GPU内存增长）
watch -n 1 nvidia-smi

# 之后的任务会启动得更快（模型已在缓存中）
```

### 2. 使用更快的启动配置

如果不需要所有组件，可以简化配置来加快启动：
- 减少 GPU 数量
- 使用更小的模型
- 禁用不需要的功能

### 3. 监控服务器启动进度

添加日志查看脚本：
```bash
# 实时查看最新的服务器日志
tail -f /workspace/logs/job_*_stderr.log
```

---

## 总结

**问题**: 客户端连接超时（3 次重试，~14 秒）  
**原因**: 服务器初始化需要更长时间（加载模型到 GPU）  
**解决**: 
1. ✅ 增加重试次数到 10 次
2. ✅ 增加重试延迟到 5 秒
3. ✅ 添加 wait_for_server.py 工具
4. ✅ 使用指数退避，最长等待 60 秒

**现在客户端最多会等待约 5 分钟**，应该足够大多数服务器完成初始化。
