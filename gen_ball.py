import blenderproc as bproc
import numpy as np
import os
import random

# ── 输出目录 ──────────────────────────────────────────────
OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), "my_data_synth")
IMG_DIR      = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR    = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(IMG_DIR,   exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────
NUM_IMAGES   = 200          # 生成图片数量
IMG_W, IMG_H = 640, 640     # 分辨率
BALL_CLASS   = 0            # YOLO 类别 ID（cat ball = 0）

# ── 1. 初始化 BlenderProc ─────────────────────────────────
bproc.init()

# ── 2. 创建球体 ───────────────────────────────────────────
sphere = bproc.object.create_primitive("SPHERE", radius=0.12)
sphere.set_cp("category_id", BALL_CLASS + 1)   # bproc 内部从 1 开始

# ── 3. 创建地面（让场景更真实，减少悬空感）─────────────────
floor = bproc.object.create_primitive("PLANE", size=10)
floor_mat = bproc.material.create("floor_mat")
floor_mat.set_principled_shader_value("Base Color", [0.4, 0.4, 0.4, 1.0])
floor.replace_materials(floor_mat)

# ── 4. 设置渲染器 ─────────────────────────────────────────
bproc.renderer.set_output_format(enable_transparency=False)
bproc.renderer.set_max_amount_of_samples(32)   # 低采样数让速度更快
bproc.camera.set_resolution(IMG_W, IMG_H)

# ── 4b. 创建点光源（循环内只更新参数）────────────────────────
light = bproc.types.Light()
light.set_type("POINT")

# ── 5. 批量渲染循环 ───────────────────────────────────────
for i in range(NUM_IMAGES):

    # -- 5a. 随机球体颜色（各种颜色让模型更泛化）
    color = [random.random(), random.random(), random.random(), 1.0]
    mat = bproc.material.create(f"mat_{i}")
    mat.set_principled_shader_value("Base Color", color)
    mat.set_principled_shader_value("Roughness", random.uniform(0.2, 0.9))
    mat.set_principled_shader_value("Metallic",  random.uniform(0.0, 0.5))
    sphere.replace_materials(mat)

    # -- 5b. 随机球体位置（XY 偏移，Z 贴地）
    bx = random.uniform(-1.5, 1.5)
    by = random.uniform(-1.5, 1.5)
    sphere.set_location([bx, by, 0.12])   # Z=radius，让球贴地

    # -- 5c. 随机相机位置（从不同角度俯视或平视球）
    cam_dist  = random.uniform(0.6, 2.5)
    cam_theta = random.uniform(0, 2 * np.pi)        # 水平方向随机
    cam_phi   = random.uniform(np.pi / 8, np.pi / 2.5)  # 仰角 ~22° – 72°
    cam_x = bx + cam_dist * np.sin(cam_phi) * np.cos(cam_theta)
    cam_y = by + cam_dist * np.sin(cam_phi) * np.sin(cam_theta)
    cam_z = cam_dist * np.cos(cam_phi) + 0.12

    cam_pose = bproc.math.build_transformation_mat(
        [cam_x, cam_y, cam_z],
        bproc.camera.rotation_from_forward_vec(
            [bx - cam_x, by - cam_y, 0.12 - cam_z]   # 看向球心
        )
    )
    bproc.camera.add_camera_pose(cam_pose)

    # -- 5d. 随机点光源（更新已有光源对象的参数）
    light.set_location([
        bx + random.uniform(-2, 2),
        by + random.uniform(-2, 2),
        random.uniform(1.5, 4.0)
    ])
    light.set_energy(random.uniform(200, 800))
    light.set_color([1.0, random.uniform(0.85, 1.0), random.uniform(0.75, 1.0)])

    # -- 5e. 渲染这一帧
    data = bproc.renderer.render()

    # -- 5f. 从分割图计算 YOLO bounding box
    seg_maps = bproc.renderer.render_segmap(map_by=["instance", "class"])
    # seg_maps["class_segmaps"][0] 是 H×W 的类别掩码
    class_mask = np.array(seg_maps["class_segmaps"][0])

    # 找球体像素（category_id=1 对应 class=1）
    ball_pixels = np.argwhere(class_mask == 1)   # shape (N, 2)  row, col

    if len(ball_pixels) > 0:
        r_min, c_min = ball_pixels.min(axis=0)
        r_max, c_max = ball_pixels.max(axis=0)

        # 转换为 YOLO 格式（归一化中心 x,y 宽高）
        cx = (c_min + c_max) / 2.0 / IMG_W
        cy = (r_min + r_max) / 2.0 / IMG_H
        bw = (c_max - c_min) / IMG_W
        bh = (r_max - r_min) / IMG_H

        # 保存图片
        import cv2
        img_rgb = data["colors"][0]          # H×W×3 uint8
        img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        img_path = os.path.join(IMG_DIR, f"synth_{i:04d}.jpg")
        cv2.imwrite(img_path, img_bgr)

        # 保存标注
        label_path = os.path.join(LABEL_DIR, f"synth_{i:04d}.txt")
        with open(label_path, "w") as f:
            f.write(f"{BALL_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"[{i+1}/{NUM_IMAGES}] 已保存: {img_path}")
    else:
        print(f"[{i+1}/{NUM_IMAGES}] 跳过（球不在视野内）")

    # 清理这帧的光源和相机，准备下一帧
    bproc.utility.reset_keyframes()

print(f"\n完成！共生成图片到: {OUTPUT_DIR}")