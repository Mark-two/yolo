#!/usr/bin/env python3
"""
fuse_depth.py — TSDF 深度融合 + 网格提取
从 captured_depth/ 读取帧，估计相机轨迹，融合深度图，输出 mesh。
"""
import open3d as o3d
import numpy as np
import cv2
import os
import json
import glob

SCRIPT_DIR = os.path.dirname(__file__)
INPUT_DIR = os.path.join(SCRIPT_DIR, "captured_depth")
OUTPUT_MESH = os.path.join(INPUT_DIR, "fused_mesh.ply")

# ── 加载内参 ──
with open(os.path.join(INPUT_DIR, "intrinsics.json")) as f:
    K = json.load(f)
intrinsics = o3d.camera.PinholeCameraIntrinsic(K["width"], K["height"], K["fx"], K["fy"], K["cx"], K["cy"])
print(f"内参: {K['width']}x{K['height']} fx={K['fx']:.1f} fy={K['fy']:.1f}")

# ── 加载帧 ──
color_files = sorted(glob.glob(os.path.join(INPUT_DIR, "frame_*.png")))
depth_files = sorted(glob.glob(os.path.join(INPUT_DIR, "depth_*.npy")))

if len(color_files) < 5:
    print(f"ERROR: 只有 {len(color_files)} 帧，至少需要 5 帧")
    exit(1)

print(f"加载 {len(color_files)} 帧")

images_rgb = []
images_depth = []
for cf, df in zip(color_files, depth_files):
    color = cv2.cvtColor(cv2.imread(cf), cv2.COLOR_BGR2RGB)
    depth = np.load(df).astype(np.float32) / 1000.0  # mm → m
    images_rgb.append(color)
    images_depth.append(depth)

# ── 估计球心位置 ──
# 对每帧取画面中心 20x20 区域的中位深度，取中位数作为球距
center_depths = []
for d in images_depth:
    h, w = d.shape
    patch = d[h//2-10:h//2+10, w//2-10:w//2+10]
    patch = patch[(patch > 0.01) & (patch < 5.0)]
    if len(patch) > 0:
        center_depths.append(np.median(patch))
center_depths.sort()
ball_dist = np.median(center_depths)
print(f"估计球距相机: {ball_dist:.3f}m")

# ── 建立初始相机轨迹（圆形） ──
N = len(color_files)
ball_radius = 0.06  # 猫玩具球约 6cm 半径，可手动调整
orbit_radius = ball_dist  # 相机到球心的水平距离约等于深度距离
orbit_height = ball_radius * 0.5  # 相机略高于球心

poses = []
for i in range(N):
    angle = 2 * np.pi * i / N
    # 相机在水平面绕球转，高度略高于球心
    cx = orbit_radius * np.cos(angle)
    cy = orbit_radius * np.sin(angle)
    cz = orbit_height
    cam_pos = np.array([cx, cy, cz])

    # 看向球心 (原点)
    forward = -cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0, 0, 1])
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = up
    T[:3, 2] = forward
    T[:3, 3] = cam_pos
    poses.append(T)

# ── ICP 逐帧精化 ──
print("ICP 精化相机位姿...")
MAX_DEPTH = 2.0
VOXEL_SIZE = 0.005

for i in range(1, N):
    # 从深度图生成点云
    d_prev = images_depth[i - 1]
    d_curr = images_depth[i]
    c_prev = images_rgb[i - 1]
    c_curr = images_rgb[i]

    # Create RGBD images
    rgbd_prev = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(c_prev),
        o3d.geometry.Image(d_prev),
        depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False
    )
    rgbd_curr = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(c_curr),
        o3d.geometry.Image(d_curr),
        depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False
    )

    # 初始变换 = 相邻帧位姿差
    init_transform = np.linalg.inv(poses[i - 1]) @ poses[i]

    # 用 colored ICP 精化
    option = o3d.pipelines.odometry.OdometryOption()
    option.max_depth = MAX_DEPTH
    option.min_depth = 0.05
    option.max_depth_diff = 0.05

    # Direct colored ICP odometry between two frames
    success, refined_T, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
        rgbd_prev, rgbd_curr,
        intrinsics, init_transform,
        o3d.pipelines.odometry.RGBDOdometryJacobianFromColor(),  # uses color info
        option
    )

    if success:
        poses[i] = poses[i - 1] @ refined_T
        if (i + 1) % 5 == 0:
            print(f"  ICP refined {i + 1}/{N}")
    else:
        print(f"  Frame {i}: ICP 失败，保留粗估计")

# ── TSDF 融合 ──
print("TSDF 深度融合...")
voxel_length = 0.002  # 2mm
sdf_trunc = 0.008  # 8mm
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=voxel_length,
    sdf_trunc=sdf_trunc,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
)

for i in range(N):
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(images_rgb[i]),
        o3d.geometry.Image(images_depth[i]),
        depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False
    )
    # Integrate: camera extrinsics = world-to-camera = inverse of camera-to-world
    volume.integrate(rgbd, intrinsics, np.linalg.inv(poses[i]))

    if (i + 1) % 10 == 0:
        print(f"  Integrated {i + 1}/{N} frames")

# ── 提取网格 ──
print("提取网格...")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()

# 去噪：保留最大连通分量
components = mesh.connected_components()
largest = max(components, key=lambda c: len(c.triangles))
mesh = largest

print(f"网格: {len(mesh.vertices):,} 顶点, {len(mesh.triangles):,} 三角面")
o3d.io.write_triangle_mesh(OUTPUT_MESH, mesh)
print(f"已保存: {OUTPUT_MESH}")
