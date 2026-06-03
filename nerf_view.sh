#!/usr/bin/env bash
# nerf_view.sh
# 启动 Nerfstudio 查看器，在浏览器中实时查看已训练的模型
#
# 用法:
#   ./nerf_view.sh [实验名]
#   ./nerf_view.sh                          # 自动加载最新模型

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="$SCRIPT_DIR/nerf_output"

# 找实验
if [ -n "$1" ]; then
    EXP_NAME="$1"
    CHECKPOINT_DIR=$(find "$OUTPUT_BASE" -name "config.yml" -path "*$EXP_NAME*" | head -1 | xargs dirname 2>/dev/null)
else
    CHECKPOINT_DIR=$(find "$OUTPUT_BASE" -name "config.yml" | xargs ls -t 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
fi

if [ -z "$CHECKPOINT_DIR" ] || [ ! -f "$CHECKPOINT_DIR/config.yml" ]; then
    echo "错误: 未找到训练结果"
    echo "请先运行 ./nerf_train.sh"
    exit 1
fi

CONFIG_FILE="$CHECKPOINT_DIR/config.yml"
echo "加载模型: $CONFIG_FILE"
echo ""
echo "在浏览器打开: http://localhost:7007"
echo "（Ctrl+C 退出）"
echo ""

conda run -n nerfstudio ns-viewer \
    --load-config "$CONFIG_FILE"
