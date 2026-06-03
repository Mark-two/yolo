#!/usr/bin/env bash
# nerf_process.sh
# 将真实拍摄的图片处理成 Nerfstudio 可用的 transforms.json 格式
# 使用 COLMAP 做相机位姿估计（比 Meshroom 更稳定）
#
# 用法:
#   ./nerf_process.sh [图片目录] [输出目录]
#   ./nerf_process.sh                          # 使用默认路径

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 默认路径 ──────────────────────────────────────────────────────────────────
INPUT_DIR="${1:-$SCRIPT_DIR/datasets/new_captures}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/nerf_data}"

echo "============================================================"
echo "  Nerfstudio 数据预处理"
echo "  输入图片目录: $INPUT_DIR"
echo "  输出 COLMAP 数据: $OUTPUT_DIR"
echo "============================================================"

# 检查图片目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 图片目录不存在: $INPUT_DIR"
    exit 1
fi

IMG_COUNT=$(ls "$INPUT_DIR"/*.{jpg,JPG,png,PNG} 2>/dev/null | wc -l)
echo "找到 $IMG_COUNT 张图片"

if [ "$IMG_COUNT" -lt 10 ]; then
    echo "警告: 图片数量较少，建议至少 30 张以上以获得好效果"
fi

mkdir -p "$OUTPUT_DIR"

# ── 使用 ns-process-data 处理（自动调用 COLMAP）───────────────────────────────
echo ""
echo "正在运行 COLMAP 相机位姿估计..."
echo "（这可能需要几分钟，请耐心等待）"
echo ""

conda run -n nerfstudio ns-process-data images \
    --data "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --camera-type perspective \
    --num-downscales 1 \
    --verbose

echo ""
echo "============================================================"
echo "  预处理完成！"
echo "  数据已保存到: $OUTPUT_DIR"
echo "  下一步: 运行 ./nerf_train.sh 开始训练"
echo "============================================================"
