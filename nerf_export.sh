#!/usr/bin/env bash
# nerf_export.sh
# 从训练好的 NeRF 模型导出点云或网格
#
# 用法:
#   ./nerf_export.sh [实验名]
#   ./nerf_export.sh ball_20260424_120000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="$SCRIPT_DIR/nerf_output"
EXPORT_DIR="$SCRIPT_DIR/nerf_export"

# 找最新的实验
if [ -n "$1" ]; then
    EXP_NAME="$1"
    # 在 nerfacto/splatfacto 子目录中查找
    CHECKPOINT_DIR=$(find "$OUTPUT_BASE" -name "config.yml" -path "*$EXP_NAME*" | head -1 | xargs dirname 2>/dev/null)
else
    # 自动找最新的训练结果
    CHECKPOINT_DIR=$(find "$OUTPUT_BASE" -name "config.yml" | xargs ls -t 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    EXP_NAME=$(basename "$(dirname "$CHECKPOINT_DIR")" 2>/dev/null)
fi

if [ -z "$CHECKPOINT_DIR" ] || [ ! -f "$CHECKPOINT_DIR/config.yml" ]; then
    echo "错误: 未找到训练结果"
    echo "请先运行 ./nerf_train.sh"
    echo "或指定实验名: ./nerf_export.sh <实验名>"
    exit 1
fi

CONFIG_FILE="$CHECKPOINT_DIR/config.yml"
mkdir -p "$EXPORT_DIR/$EXP_NAME"

echo "============================================================"
echo "  Nerfstudio 模型导出"
echo "  配置文件: $CONFIG_FILE"
echo "  导出目录: $EXPORT_DIR/$EXP_NAME"
echo "============================================================"

# 检测是 NeRF 还是 Gaussian Splatting
if grep -q "splatfacto" "$CONFIG_FILE" 2>/dev/null; then
    echo ""
    echo "检测到 Gaussian Splatting 模型，导出 .ply 点云..."
    conda run -n nerfstudio ns-export gaussian-splat \
        --load-config "$CONFIG_FILE" \
        --output-dir "$EXPORT_DIR/$EXP_NAME"
else
    echo ""
    echo "检测到 NeRF 模型，导出点云 + 网格..."
    
    # 1. 导出点云
    echo "步骤 1/2: 导出点云..."
    conda run -n nerfstudio ns-export pointcloud \
        --load-config "$CONFIG_FILE" \
        --output-dir "$EXPORT_DIR/$EXP_NAME" \
        --num-points 1000000 \
        --remove-outliers True \
        --normal-method model_output

    # 2. 导出网格（Poisson 重建）
    echo "步骤 2/2: 从点云重建网格..."
    conda run -n nerfstudio ns-export marching-cubes \
        --load-config "$CONFIG_FILE" \
        --output-dir "$EXPORT_DIR/$EXP_NAME" \
        --resolution 512
fi

echo ""
echo "============================================================"
echo "  导出完成！"
echo "  文件保存在: $EXPORT_DIR/$EXP_NAME/"
ls "$EXPORT_DIR/$EXP_NAME/" 2>/dev/null
echo "============================================================"
