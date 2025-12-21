# CORS 错误修复说明

## 问题

Web Dashboard 显示错误：
```
"OPTIONS /list_jobs HTTP/1.1" 405 Method Not Allowed
```

## 原因

这是一个 **CORS (跨域资源共享)** 问题：

1. Web Dashboard 在浏览器中运行（`http://localhost:8081`）
2. 尝试访问调度器 API（`http://localhost:8765`）
3. 由于端口不同，浏览器认为这是跨域请求
4. 当请求包含 `Authorization` 头时，浏览器会先发送 OPTIONS 预检请求
5. 调度器没有配置 CORS，拒绝了 OPTIONS 请求

## 解决方案

已在调度器中添加 CORS 中间件支持！

### 修改内容

文件：`scheduler/job_scheduler.py`

1. 添加 CORS 中间件导入
2. 配置允许所有来源的跨域请求
3. 允许所有 HTTP 方法（包括 OPTIONS）
4. 允许所有请求头（包括 Authorization）

### 应用修复

**重启调度器**：

```bash
# 1. 停止当前运行的调度器（Ctrl+C）

# 2. 重新启动调度器
python scheduler/launch_scheduler.py \
    available_gpus=[0,1,2,3] \
    scheduler_port=8765
```

### 验证修复

1. **启动 Web Dashboard**：
   ```bash
   python scheduler/web_dashboard.py --port 8081
   ```

2. **刷新浏览器**：打开 `http://localhost:8081/web_dashboard.html`

3. **输入 API Key** 并保存

4. **检查结果**：
   - ✅ 应该能看到任务列表
   - ✅ 不再有 405 错误
   - ✅ OPTIONS 请求成功返回 200

### 技术细节

添加的 CORS 配置：
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 允许所有来源
    allow_credentials=True,   # 允许携带凭证
    allow_methods=["*"],      # 允许所有方法
    allow_headers=["*"],      # 允许所有请求头
)
```

**生产环境注意**：
在生产环境中，应该限制 `allow_origins` 为特定域名：
```python
allow_origins=["https://your-dashboard-domain.com"]
```

### 完整流程

现在 Web Dashboard 的完整工作流程：

1. 🌐 **浏览器**：打开 Dashboard → `http://localhost:8081/web_dashboard.html`
2. 🔑 **输入 API Key**：保存到 localStorage
3. 📡 **OPTIONS 请求**：浏览器发送预检请求 → 调度器允许
4. 📊 **GET 请求**：带 Authorization 头请求数据 → 调度器返回任务列表
5. ✅ **显示数据**：Dashboard 显示所有任务

全部流程现在都能正常工作！
