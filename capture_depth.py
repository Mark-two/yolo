#!/usr/bin/env python3
"""
capture_depth.py — RealSense D435i 深度捕获
用法：每次绕球拍一张，按 S 保存一帧，按 Q 退出。
输出：captured_depth/ 下 frame_0000.png + depth_0000.npy
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import json

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "captured_depth")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 启动 RealSense ──
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# 对齐深度到彩色
align = rs.align(rs.stream.color)

# 获取内参
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K = {
    "width": intr.width, "height": intr.height,
    "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
    "model": intr.model.name,
}
with open(os.path.join(OUTPUT_DIR, "intrinsics.json"), "w") as f:
    json.dump(K, f, indent=2)
print(f"内参已保存: {K['width']}x{K['height']} fx={K['fx']:.2f} fy={K['fy']:.2f}")

frame_idx = 0
print(f"\n按 S 保存当前帧 | 按 Q 退出")
print(f"输出目录: {OUTPUT_DIR}")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_img = np.asanyarray(depth_frame.get_data())  # uint16, mm
        color_img = np.asanyarray(color_frame.get_data())

        # 显示
        display = color_img.copy()
        cv2.putText(display, f"Frames saved: {frame_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Depth Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png"), color_img)
            np.save(os.path.join(OUTPUT_DIR, f"depth_{frame_idx:04d}.npy"), depth_img)
            print(f"  Saved frame {frame_idx}")
            frame_idx += 1
        elif key == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print(f"共保存 {frame_idx} 帧到 {OUTPUT_DIR}/")
print("下一步: python fuse_depth.py")
