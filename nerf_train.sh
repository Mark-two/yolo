#!/usr/bin/env bash
# nerf_train.sh
# 使用 Nerfstudio 训练 NeRF / 3D Gaussian Splatting 模型
#
# 支持的模型:
#   nerfacto      — 推荐，实时训练真实场景（约1分钟可见结果）
#   splatfacto    — 3D Gaussian Splatting，渲染更快，质量更高
#   instant-ngp   — 极速（秒级），质量稍低
#
# 用法:
#   ./nerf_train.sh [模型] [数据目录] [实验名]
#   ./nerf_train.sh nerfacto                  # 使用默认路径
#   ./nerf_train.sh splatfacto ./nerf_data my_ball

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:-nerfacto}"
DATA_DIR="${2:-$SCRIPT_DIR/nerf_data}"
EXP_NAME="${3:-ball_$(date +%Y%m%d_%H%M%S)}"

echo "============================================================"
echo "  Nerfstudio 训练"
echo "  模型:     $MODEL"
echo "  数据目录:  $DATA_DIR"
echo "  实验名:    $EXP_NAME"
echo "============================================================"

# 检查数据目录和 transforms.json
if [ ! -f "$DATA_DIR/transforms.json" ]; then
    echo "错误: 未找到 $DATA_DIR/transforms.json"
    echo "请先运行 ./nerf_process.sh 进行数据预处理"
    exit 1
fi

echo ""
echo "开始训练 $MODEL ..."
echo "训练过程中可在浏览器打开 http://localhost:7007 查看实时进度"
echo ""

# ── 根据模型选择参数 ──────────────────────────────────────────────────────────
if [ "$MODEL" = "splatfacto" ]; then
    # 3D Gaussian Splatting
    conda run -n nerfstudio ns-train splatfacto \
        --data "$DATA_DIR" \
        --experiment-name "$EXP_NAME" \
        --output-dir "$SCRIPT_DIR/nerf_output" \
        --max-num-iterations 30000 \
        nerfstudio-data \
            --downscale-factor 1

elif [ "$MODEL" = "instant-ngp" ]; then
    # 极速模式
    conda run -n nerfstudio ns-train instant-ngp \
        --data "$DATA_DIR" \
        --experiment-name "$EXP_NAME" \
        --output-dir "$SCRIPT_DIR/nerf_output" \
        nerfstudio-data \
            --downscale-factor 2

else
    # 默认 nerfacto（实拍最推荐）
    conda run -n nerfstudio ns-train nerfacto \
        --data "$DATA_DIR" \
        --experiment-name "$EXP_NAME" \
        --output-dir "$SCRIPT_DIR/nerf_output" \
        --max-num-iterations 30000 \
        --pipeline.model.predict-normals True \
        nerfstudio-data \
            --downscale-factor 1
fi

echo ""
echo "============================================================"
echo "  训练完成！"
echo "  模型保存在: $SCRIPT_DIR/nerf_output/$EXP_NAME/"
echo "  下一步:"
echo "    查看结果:   ./nerf_view.sh $EXP_NAME"
echo "    导出网格:   ./nerf_export.sh $EXP_NAME"
echo "============================================================"
