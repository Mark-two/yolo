#!/usr/bin/env python3
"""
sam_segment.py — SAM2 全自动抠图 → Meshroom 三维重建 管线

工作流：
  1. 从 INPUT_DIR 读取所有照片（机器人拍的 100 张）
  2. 以画面中心点为 foreground prompt，调用 SAM2 分割中心物体
  3. 把背景涂成纯黑色（或透明 alpha），保存到 OUTPUT_DIR
     Meshroom 对黑色像素无法建特征点，自然忽略背景
  4. （可选）用 --meshroom 参数自动调用 meshroom_batch 启动三维重建

用法示例：
  # 基本：处理 datasets/new_captures/ 中的照片
  python sam_segment.py

  # 指定输入目录
  python sam_segment.py --input my_photos/

  # 透明背景 PNG（比黑背景更干净，但 Meshroom 需支持 PNG）
  python sam_segment.py --alpha

  # 抠图后自动启动 Meshroom
  python sam_segment.py --meshroom

  # 完整参数
  python sam_segment.py --input datasets/new_captures/ \\
                         --output sam_masked/ \\
                         --model sam2.1_b+.pt \\
                         --meshroom \\
                         --meshroom-out meshroom_real_output/

SAM2 模型大小（从小到大，自动下载）：
  sam2.1_t.pt  — Tiny   (~38 MB，最快)
  sam2.1_s.pt  — Small  (~46 MB)
  sam2.1_b.pt  — Base   (~80 MB，推荐)
  sam2.1_l.pt  — Large  (~224 MB，最精确)
"""

import os
import glob
import shutil
import subprocess
import argparse
from pathlib import Path

import numpy as np
import cv2

# ─────────────────────── 配置区（直接修改此处即可）───────────────────────
INPUT_DIR   = "datasets/new_captures"  # 机器人拍的原始照片目录
OUTPUT_DIR  = "sam_masked"             # 抠图后输出目录（喂给 Meshroom）
SAM_MODEL   = "weights/sam2.1_b.pt"   # SAM2 模型，首次运行自动下载
BLACK_BG    = True                     # True=纯黑背景 JPEG，False=透明 PNG
# ─────────────────────────────────────────────────────────────────────────


# ─────────────────────────── 核心：选最佳 Mask ───────────────────────────
def pick_best_mask(masks_np: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    从 SAM2 返回的多个候选 mask 中，选出最像"画面中心单个物体"的一个。

    策略（按优先级）：
      A. 过滤掉面积 < 0.5% 或 > 90% 的 mask（噪声 / 整幅图）
      B. 在过滤后的 mask 里，优先选"中心点在 mask 内"的候选
      C. 若有多个含中心点的候选，选占比最接近 20% 的
         （物体通常占镜头的 15-30%，这是最稳的经验值）
      D. 若中心点不在任何 mask 里（极端情况），退化为选最接近 20% 的最大 mask

    Args:
        masks_np : shape (N, H, W), float32，取值 [0, 1]
        h, w     : 图像高宽（像素）

    Returns:
        (H, W) bool ndarray，True = 物体区域
    """
    total_pixels = h * w
    cx, cy = w // 2, h // 2

    valid_masks = []
    for m in masks_np:
        m_bool = m > 0.5
        ratio = m_bool.sum() / total_pixels
        if 0.005 < ratio < 0.90:
            has_center = bool(m_bool[cy, cx])
            valid_masks.append((m_bool, ratio, has_center))

    if not valid_masks:
        # SAM2 返回了 0 个 mask（极端情况），输出全黑（Meshroom 会跳过该帧）
        if len(masks_np) == 0:
            return np.zeros((h, w), dtype=bool)
        # 所有 mask 都被面积过滤掉，兜底：选面积最大的那个
        areas = [np.sum(m > 0.5) for m in masks_np]
        return masks_np[np.argmax(areas)] > 0.5

    # 优先选含中心点的候选
    center_masks = [(m, r) for m, r, c in valid_masks if c]
    pool = center_masks if center_masks else [(m, r) for m, r, _ in valid_masks]

    # 选占比最接近 20% 的（对应"前景物体"尺度）
    TARGET_RATIO = 0.20
    pool.sort(key=lambda x: abs(x[1] - TARGET_RATIO))
    return pool[0][0]


# ─────────────────────────── 步骤 1：批量抠图 ────────────────────────────
def process_images(
    input_dir: str,
    output_dir: str,
    model_name: str,
    black_bg: bool,
    save_debug: bool = False,
) -> int:
    """
    遍历 input_dir 里的所有图片，用 SAM2 进行中心物体分割，
    输出黑背景 JPEG（或透明 PNG）到 output_dir。

    Returns:
        成功处理的图片数量
    """
    from ultralytics import SAM  # 延迟导入，避免无 GPU 时启动失败

    os.makedirs(output_dir, exist_ok=True)
    if save_debug:
        os.makedirs(os.path.join(output_dir, "_debug"), exist_ok=True)

    # 收集图片路径（去重、排序）
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    image_paths = sorted(set(image_paths))

    if not image_paths:
        raise FileNotFoundError(
            f"在 {input_dir!r} 中未找到任何图片（jpg / jpeg / png）。\n"
            "  请确认路径正确，或用 --input 指定正确的照片目录。"
        )

    print(f"\n[SAM] 模型  : {model_name}")
    print(f"[SAM] 输入  : {os.path.abspath(input_dir)}  ({len(image_paths)} 张)")
    print(f"[SAM] 输出  : {os.path.abspath(output_dir)}")
    print(f"[SAM] 背景  : {'纯黑 JPEG' if black_bg else '透明 PNG'}")
    print()

    model = SAM(model_name)  # 首次运行时自动下载权重

    ok_count = 0
    for i, img_path in enumerate(image_paths, 1):
        stem = Path(img_path).stem
        out_ext = ".jpg" if black_bg else ".png"
        out_path = os.path.join(output_dir, stem + out_ext)

        prefix = f"[{i:3d}/{len(image_paths)}] {os.path.basename(img_path)}"
        print(prefix, end=" ... ", flush=True)

        # ── 读取原图 ──
        img = cv2.imread(img_path)
        if img is None:
            print("⚠️  读取失败，跳过")
            continue

        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # ── 读取原图 EXIF（卡口焦距等，Meshroom SfM 依赖这些信息）──
        exif_bytes = None
        try:
            import piexif
            exif_data = piexif.load(img_path)
            exif_bytes = piexif.dump(exif_data)
        except Exception:
            pass  # 非 JPEG 或无 EXIF，忽略

        # ── 运行 SAM2：以图像中心为 foreground 点 prompt ──
        results = model(
            img_path,
            points=[[cx, cy]],   # 前景点坐标（图像像素坐标）
            labels=[1],          # 1 = foreground
            verbose=False,
        )

        masks_obj = results[0].masks if (results and len(results) > 0) else None
        if (
            results is None
            or len(results) == 0
            or masks_obj is None
            or len(masks_obj.data) == 0
        ):
            # 无法分割：输出全黑图（Meshroom 会自动忽略）
            print("⚠️  无 mask，输出全黑图（继续）")
            out_img = np.zeros_like(img)
            cv2.imwrite(out_path, out_img)
            continue

        # ── 取出所有 mask  (N, H, W) ──
        masks_np = masks_obj.data.cpu().numpy().astype(np.float32)

        # SAM2 返回的 mask 尺寸有时与原图不同，需要 resize
        if masks_np.shape[1:] != (h, w):
            masks_np = np.stack([
                cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                for m in masks_np
            ])

        # ── 挑选最佳 mask ──
        best_mask = pick_best_mask(masks_np, h, w)
        ratio_pct = best_mask.sum() / (h * w) * 100

        # ── 应用 mask，生成输出图 ──
        if black_bg:
            out_img = img.copy()
            out_img[~best_mask] = 0  # 背景涂黑
            cv2.imwrite(out_path, out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # 把原 EXIF 写回（保留相机焦距，Meshroom 需要）
            if exif_bytes:
                try:
                    import piexif
                    piexif.insert(exif_bytes, out_path)
                except Exception:
                    pass
        else:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            rgba[~best_mask, 3] = 0  # 背景透明
            cv2.imwrite(out_path, rgba)

        # ── （可选）调试：保存可视化 mask 叠加图 ──
        if save_debug:
            vis = img.copy()
            overlay = np.zeros_like(img)
            overlay[best_mask] = [0, 255, 0]  # 绿色前景
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)  # 红点标注中心
            debug_path = os.path.join(output_dir, "_debug", stem + "_mask.jpg")
            cv2.imwrite(debug_path, vis)

        ok_count += 1
        print(f"✅  物体占比 {ratio_pct:5.1f}%")

    print(
        f"\n[SAM] 完成！成功 {ok_count} / {len(image_paths)} 张\n"
        f"[SAM] 抠图结果：{os.path.abspath(output_dir)}\n"
    )
    return ok_count


# ─────────────────────── 步骤 2：自动启动 Meshroom ───────────────────────
def launch_meshroom(image_dir: str, meshroom_out: str) -> None:
    """
    调用 meshroom_batch（命令行版 Meshroom）进行 SfM 三维重建。

    前提：
      - Meshroom 已安装（AppImage 解压 / 系统包），
        且 meshroom_batch 在 PATH 中可用。
      - 图片有正确的 EXIF 焦距信息（sam_segment 已自动保留）。

    重建耗时：
      100 张 1280×720 图片约需 30~90 分钟（取决于 GPU 是否可用）。
    """
    exts = ("*.jpg", "*.jpeg", "*.png")
    images = []
    for ext in exts:
        images.extend(glob.glob(os.path.join(image_dir, ext)))
    images = sorted(set(images))

    if not images:
        print(f"[Meshroom] ⚠️  在 {image_dir!r} 中未找到图片，跳过重建")
        return

    os.makedirs(meshroom_out, exist_ok=True)
    print(f"[Meshroom] 启动三维重建...")
    print(f"  输入 : {len(images)} 张抠图后的照片")
    print(f"  输出 : {os.path.abspath(meshroom_out)}")
    print(f"  （耗时可能较长，请耐心等待）\n")

    cmd = (
        ["meshroom_batch", "--input"]
        + images
        + ["--output", meshroom_out, "--save", os.path.join(meshroom_out, "project.mg")]
    )

    try:
        subprocess.run(cmd, check=True)
        print(f"\n[Meshroom] 三维重建完成！")
        print(f"  OBJ / MTL 文件在：{os.path.abspath(meshroom_out)}")
        print(
            "  下一步：运行 extract_only.py（blenderproc）把底部平面切掉，"
            "再用 bmesh 封口 + 灰材质收尾。"
        )
    except FileNotFoundError:
        print(
            "[Meshroom] ❌  未找到 meshroom_batch 命令。\n"
            "  请检查 Meshroom 是否已正确安装并添加到 PATH。\n"
            "  或者：手动把抠图目录拖进 Meshroom GUI 界面运行。\n"
            f"  抠图目录：{os.path.abspath(image_dir)}"
        )
    except subprocess.CalledProcessError as e:
        print(f"[Meshroom] ❌  重建失败（exit code {e.returncode}）")


# ─────────────────────────────── 入口 ────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SAM2 全自动抠图 → Meshroom 三维重建 管线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default=INPUT_DIR,
        metavar="DIR",
        help=f"原始照片目录（默认：{INPUT_DIR!r}）",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_DIR,
        metavar="DIR",
        help=f"抠图输出目录（默认：{OUTPUT_DIR!r}）",
    )
    parser.add_argument(
        "--model", "-m",
        default=SAM_MODEL,
        metavar="MODEL",
        help=(
            f"SAM2 模型名（默认：{SAM_MODEL!r}）\n"
            "  可选：sam2.1_t.pt / sam2.1_s.pt / sam2.1_b.pt / sam2.1_l.pt"
        ),
    )
    parser.add_argument(
        "--alpha", "-a",
        action="store_true",
        help="输出透明背景 PNG（默认：黑背景 JPEG）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="同时在 OUTPUT/_debug/ 保存 mask 可视化叠加图（便于调参）",
    )
    parser.add_argument(
        "--meshroom",
        action="store_true",
        help="抠图完成后自动调用 meshroom_batch 进行三维重建",
    )
    parser.add_argument(
        "--meshroom-out",
        default="meshroom_real_output",
        metavar="DIR",
        help="Meshroom 输出目录（默认：'meshroom_real_output'）",
    )
    args = parser.parse_args()

    # ── 步骤 1：SAM2 批量抠图 ──────────────────────────────────────────
    ok = process_images(
        input_dir=args.input,
        output_dir=args.output,
        model_name=args.model,
        black_bg=not args.alpha,
        save_debug=args.debug,
    )

    if ok == 0:
        print("⚠️  没有任何图片处理成功，中止流程。")
        return

    # ── 步骤 2：（可选）自动启动 Meshroom ─────────────────────────────
    if args.meshroom:
        launch_meshroom(args.output, args.meshroom_out)
    else:
        print("提示：加 --meshroom 参数可在抠图结束后自动启动 Meshroom 三维重建。")
        print(
            f"  或者：手动将 {os.path.abspath(args.output)} 目录"
            " 拖入 Meshroom GUI → 点击 Start。"
        )


if __name__ == "__main__":
    main()
