import blenderproc as bproc
import bpy
import numpy as np
import os
import math
import piexif

# ── 输出目录 ──────────────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "my_data_meshroom")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 参数 ──────────────────────────────────────────────────
IMG_W, IMG_H = 1920, 1080   # 高分辨率，Meshroom 效果更好
BALL_RADIUS  = 1.2    # 放大10倍，避免 Meshroom bounding box 过小

# ── 1. 初始化 ─────────────────────────────────────────────
bproc.init()

# ── 2. 球体 + 程序化花纹贴图（所有帧保持一致！）────────────
sphere = bproc.object.create_primitive("SPHERE", radius=BALL_RADIUS)
sphere.set_location([0, 0, BALL_RADIUS])  # 贴地放置

# 用 bpy 直接建节点树：Voronoi 纹理 → 彩色渐变
ball_mat = bpy.data.materials.new("ball_mat")
ball_mat.use_nodes = True
nt = ball_mat.node_tree
nt.nodes.clear()

bsdf  = nt.nodes.new('ShaderNodeBsdfPrincipled')
out   = nt.nodes.new('ShaderNodeOutputMaterial')
coord = nt.nodes.new('ShaderNodeTexCoord')
voron = nt.nodes.new('ShaderNodeTexVoronoi')
ramp  = nt.nodes.new('ShaderNodeValToRGB')

voron.inputs['Scale'].default_value = 7.0
bsdf.inputs['Roughness'].default_value = 0.35

# 四色渐变（模拟真实猫玩具球的花纹）
ramp.color_ramp.elements.new(0.33)
ramp.color_ramp.elements.new(0.66)
ramp.color_ramp.elements[0].color = (0.9, 0.1, 0.1, 1.0)  # 红
ramp.color_ramp.elements[1].color = (0.9, 0.75, 0.05, 1.0) # 黄
ramp.color_ramp.elements[2].color = (0.1, 0.35, 0.9, 1.0)  # 蓝
ramp.color_ramp.elements[3].color = (0.1, 0.75, 0.2, 1.0)  # 绿

nt.links.new(coord.outputs['Generated'], voron.inputs['Vector'])
nt.links.new(voron.outputs['Color'],     ramp.inputs['Fac'])
nt.links.new(ramp.outputs['Color'],      bsdf.inputs['Base Color'])
nt.links.new(bsdf.outputs['BSDF'],       out.inputs['Surface'])

sphere.blender_obj.data.materials.clear()
sphere.blender_obj.data.materials.append(ball_mat)

# ── 3. 地面：棋盘格（给 Meshroom 提供大量特征点）────────────
floor = bproc.object.create_primitive("PLANE", size=30.0)
floor.set_location([0, 0, 0])

floor_mat = bpy.data.materials.new("floor_mat")
floor_mat.use_nodes = True
ft = floor_mat.node_tree
ft.nodes.clear()

fbsdf   = ft.nodes.new('ShaderNodeBsdfPrincipled')
fout    = ft.nodes.new('ShaderNodeOutputMaterial')
fcoord  = ft.nodes.new('ShaderNodeTexCoord')
checker = ft.nodes.new('ShaderNodeTexChecker')

checker.inputs['Scale'].default_value   = 10.0
checker.inputs['Color1'].default_value  = (0.95, 0.95, 0.95, 1.0)
checker.inputs['Color2'].default_value  = (0.08, 0.08, 0.08, 1.0)
fbsdf.inputs['Roughness'].default_value = 0.85

ft.links.new(fcoord.outputs['Generated'], checker.inputs['Vector'])
ft.links.new(checker.outputs['Color'],    fbsdf.inputs['Base Color'])
ft.links.new(fbsdf.outputs['BSDF'],       fout.inputs['Surface'])

floor.blender_obj.data.materials.clear()
floor.blender_obj.data.materials.append(floor_mat)

# ── 3b. 地面周围加6个彩色参考标记（给 SfM 提供稳定跨帧特征点）──
marker_configs = [
    ([  4.0,  0.0, 0.0], [0.9, 0.1, 0.1, 1.0]),  # 红
    ([ -4.0,  0.0, 0.0], [0.1, 0.8, 0.1, 1.0]),  # 绿
    ([  0.0,  4.0, 0.0], [0.1, 0.1, 0.9, 1.0]),  # 蓝
    ([  0.0, -4.0, 0.0], [0.9, 0.8, 0.0, 1.0]),  # 黄
    ([  2.8,  2.8, 0.0], [0.9, 0.4, 0.0, 1.0]),  # 橙
    ([ -2.8, -2.8, 0.0], [0.6, 0.0, 0.9, 1.0]),  # 紫
]
for pos, color in marker_configs:
    marker = bproc.object.create_primitive("CUBE", size=0.6)
    marker.set_location([pos[0], pos[1], 0.3])
    mm = bproc.material.create(f"marker_{pos[0]}")
    mm.set_principled_shader_value("Base Color", color)
    mm.set_principled_shader_value("Roughness", 0.3)
    marker.replace_materials(mm)
light_configs = [
    ([15.0, -10.0, 20.0], 35000),   # 主光
    ([-15.0,  5.0, 15.0], 17500),  # 补光
    ([  0.0, 20.0, 10.0], 12000),  # 背光
]
for pos, energy in light_configs:
    lg = bproc.types.Light()
    lg.set_type("POINT")
    lg.set_location(pos)
    lg.set_energy(energy)

# ── 5. 相机设置 ───────────────────────────────────────────
bproc.camera.set_resolution(IMG_W, IMG_H)
# 模拟手机主摄参数（等效焦距约 27mm）
bpy.context.scene.camera.data.lens        = 27
bpy.context.scene.camera.data.sensor_width = 36

# ── 6. 生成轨道相机位置 ───────────────────────────────────
#   6 层仰角 × 每层均匀水平分布 + 顶部密集补拍
#   目标：~90 张，球体顶部极点完整覆盖

elevations        = [10,  25,  40,  55,  70,  85]
positions_per_row = [16,  14,  12,   8,   8,   8]
cam_dist = 5.2    # 距球心的距离（放大10倍）

total = 0
for elev_deg, n in zip(elevations, positions_per_row):
    elev = math.radians(elev_deg)
    for j in range(n):
        azim = 2 * math.pi * j / n
        cx = cam_dist * math.cos(elev) * math.cos(azim)
        cy = cam_dist * math.cos(elev) * math.sin(azim)
        cz = cam_dist * math.sin(elev) + BALL_RADIUS  # 高度偏移贴地
        forward = [0 - cx, 0 - cy, BALL_RADIUS - cz]
        pose = bproc.math.build_transformation_mat(
            [cx, cy, cz],
            bproc.camera.rotation_from_forward_vec(forward)
        )
        bproc.camera.add_camera_pose(pose)
        total += 1

# 顶部补拍：16 个近距离高仰角相机（r 缩小让仰角接近正上方）
for azim_deg in range(0, 360, 22):
    azim = math.radians(azim_deg)
    r = 0.3   # 极小水平偏移，几乎正上方
    cx, cy, cz = r * math.cos(azim), r * math.sin(azim), 5.1
    forward = [0 - cx, 0 - cy, BALL_RADIUS - cz]
    pose = bproc.math.build_transformation_mat(
        [cx, cy, cz],
        bproc.camera.rotation_from_forward_vec(forward)
    )
    bproc.camera.add_camera_pose(pose)
    total += 1

# 正上方垂直俯视（直接补全极点）
pose = bproc.math.build_transformation_mat(
    [0.001, 0.001, BALL_RADIUS + 5.2],   # 微小偏移避免奇点
    bproc.camera.rotation_from_forward_vec([0, 0, -1])
)
bproc.camera.add_camera_pose(pose)
total += 1

print(f"共 {total} 个相机位置，开始渲染（分辨率 {IMG_W}×{IMG_H}）...")

# ── 7. 渲染所有帧 ─────────────────────────────────────────
bproc.renderer.set_max_amount_of_samples(64)
data = bproc.renderer.render()

# ── 8. 保存为 JPEG ────────────────────────────────────────
# 优先用 PIL，Blender 环境内通常已随 BlenderProc 安装
try:
    from PIL import Image
    def _save(arr, path):
        Image.fromarray(arr).save(path, quality=95)
except ImportError:
    import cv2
    def _save(arr, path):    meshroom_batch --input my_data_meshroom/ --output meshroom_synth_output/
        cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

import random as _random
def _add_noise_and_save(arr, path):
    """加高斯噪点 + 暗角，模拟真实相机，让 SIFT 特征更稳定。"""
    import cv2
    img = cv2.cvtColor(np.array(arr), cv2.COLOR_RGB2BGR).astype(np.float32)
    # 高斯噪点
    noise = np.random.normal(0, 3.5, img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)
    # 暗角
    h, w = img.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.35 * dist, 0.5, 1.0)
    img *= vignette[:, :, np.newaxis]
    cv2.imwrite(path, img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92])

def _insert_exif(path):
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make:  b"Blender",
            piexif.ImageIFD.Model: b"VirtualCamera",
        },
        "Exif": {
            piexif.ExifIFD.FocalLength:           (27, 1),
            piexif.ExifIFD.FocalLengthIn35mmFilm:  27,
            piexif.ExifIFD.PixelXDimension:        IMG_W,
            piexif.ExifIFD.PixelYDimension:        IMG_H,
        }
    }
    piexif.insert(piexif.dump(exif_dict), path)

for idx, img_rgb in enumerate(data["colors"]):
    path = os.path.join(OUTPUT_DIR, f"frame_{idx:04d}.jpg")
    _add_noise_and_save(np.array(img_rgb), path)
    _insert_exif(path)
    print(f"[{idx+1}/{total}] {path}")

print(f"\n完成！{total} 张图保存到: {OUTPUT_DIR}")
print("把整个文件夹拖入 Meshroom → 点击 Start 即可重建。")
