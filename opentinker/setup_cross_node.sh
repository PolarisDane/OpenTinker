#!/bin/bash
# 跨节点配置快速设置脚本
# 用法: ./setup_cross_node.sh <scheduler_ip> <environment_ip>

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查参数
if [ $# -lt 2 ]; then
    echo -e "${RED}错误: 参数不足${NC}"
    echo "用法: $0 <scheduler_ip> <environment_ip> [scheduler_port] [env_port]"
    echo ""
    echo "示例:"
    echo "  $0 192.168.1.100 192.168.1.101"
    echo "  $0 192.168.1.100 192.168.1.101 8766 8084"
    exit 1
fi

SCHEDULER_IP=$1
ENV_IP=$2
SCHEDULER_PORT=${3:-8766}  # 默认 8766
ENV_PORT=${4:-8084}         # 默认 8084

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}OpenTinker 跨节点配置工具${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}配置信息:${NC}"
echo "  Scheduler: http://${SCHEDULER_IP}:${SCHEDULER_PORT}"
echo "  Environment: http://${ENV_IP}:${ENV_PORT}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLIENT_CONFIG_DIR="${SCRIPT_DIR}/client/client_config"

# 检查配置目录是否存在
if [ ! -d "$CLIENT_CONFIG_DIR" ]; then
    echo -e "${RED}错误: 配置目录不存在: ${CLIENT_CONFIG_DIR}${NC}"
    echo "确保脚本位于 OpenTinker 项目根目录"
    exit 1
fi

# 配置文件列表
CONFIG_FILES=(
    "generic_env_param.yaml"
    "gomoku_param.yaml"
)

# 备份并修改配置文件
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    CONFIG_PATH="${CLIENT_CONFIG_DIR}/${CONFIG_FILE}"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo -e "${YELLOW}跳过: ${CONFIG_FILE} (文件不存在)${NC}"
        continue
    fi
    
    echo -e "${GREEN}处理: ${CONFIG_FILE}${NC}"
    
    # 创建备份
    BACKUP_PATH="${CONFIG_PATH}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$CONFIG_PATH" "$BACKUP_PATH"
    echo "  ✓ 备份已创建: $(basename $BACKUP_PATH)"
    
    # 修改 scheduler_url
    sed -i.tmp "s|scheduler_url:.*|scheduler_url: \"http://${SCHEDULER_IP}:${SCHEDULER_PORT}\"|" "$CONFIG_PATH"
    
    # 修改 env_endpoint
    sed -i.tmp "s|env_endpoint:.*|env_endpoint: \"http://${ENV_IP}:${ENV_PORT}\"  # Modified by setup script|" "$CONFIG_PATH"
    
    # 删除临时文件
    rm -f "${CONFIG_PATH}.tmp"
    
    echo "  ✓ 配置已更新"
    echo ""
done

# 验证修改
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置验证${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    CONFIG_PATH="${CLIENT_CONFIG_DIR}/${CONFIG_FILE}"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        continue
    fi
    
    echo -e "${YELLOW}${CONFIG_FILE}:${NC}"
    
    # 显示 scheduler_url
    SCHEDULER_LINE=$(grep "scheduler_url:" "$CONFIG_PATH" | head -1)
    echo "  ${SCHEDULER_LINE}"
    
    # 显示 env_endpoint
    ENV_LINE=$(grep "env_endpoint:" "$CONFIG_PATH" | head -1)
    echo "  ${ENV_LINE}"
    echo ""
done

# 网络连通性测试
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}网络连通性测试${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

echo -e "${YELLOW}测试 Scheduler 连接...${NC}"
if ping -c 1 -W 2 "$SCHEDULER_IP" &> /dev/null; then
    echo -e "  ${GREEN}✓ Ping ${SCHEDULER_IP} 成功${NC}"
else
    echo -e "  ${RED}✗ Ping ${SCHEDULER_IP} 失败${NC}"
fi

echo ""
echo -e "${YELLOW}测试 Environment 连接...${NC}"
if ping -c 1 -W 2 "$ENV_IP" &> /dev/null; then
    echo -e "  ${GREEN}✓ Ping ${ENV_IP} 成功${NC}"
else
    echo -e "  ${RED}✗ Ping ${ENV_IP} 失败${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}配置完成!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "下一步:"
echo "  1. 启动 Scheduler (在 ${SCHEDULER_IP} 节点):"
echo "     cd scheduler && python launch_scheduler.py"
echo ""
echo "  2. 启动 Environment Server (在 ${ENV_IP} 节点):"
echo "     cd environment/example && python mock_env_server.py --port ${ENV_PORT}"
echo ""
echo "  3. 运行 Client (在当前节点):"
echo "     cd client && python generic_env_client.py"
echo ""
echo -e "${YELLOW}注意: 记得在 generic_env_param.yaml 中填写正确的 scheduler_api_key!${NC}"
