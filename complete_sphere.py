import blenderproc as bproc
"""
complete_sphere.py
将 Meshroom 重建的半球沿底面赤道镜像，生成完整球体。
上半：保留原始 Meshroom 纹理
下半：镜像几何 + 灰色材质（表示未扫描区域）

用法：
  blenderproc run complete_sphere.py
"""
import numpy as np
import os
import bpy
import bmesh
from mathutils import Vector

INPUT_OBJ  = os.path.join(os.path.dirname(__file__),
                          "meshroom_real_output", "extracted.obj")
OUTPUT_OBJ = os.path.join(os.path.dirname(__file__),
                          "meshroom_real_output", "completed.obj")

# ── 初始化 ────────────────────────────────────────────────
bproc.init()

print("Loading mesh...")
objs = bproc.loader.load_obj(INPUT_OBJ)
top_obj = objs[0]
blender_obj = top_obj.blender_obj
mesh = blender_obj.data

# ── 分析当前网格范围 ──────────────────────────────────────
pts = np.array([[v.co.x, v.co.y, v.co.z] for v in mesh.vertices])
z_min = float(pts[:, 2].min())
z_max = float(pts[:, 2].max())
print(f"上半球 Z 范围: [{z_min:.4f}, {z_max:.4f}]")
print(f"球高度: {z_max - z_min:.4f} m")

# 赤道 = 当前网格的底面（被切平的那个面）
z_equator = z_min
print(f"赤道平面 Z = {z_equator:.4f}")

# ── 为灰色下半球准备材质 ──────────────────────────────────
gray_mat = bpy.data.materials.new(name="bottom_gray")
gray_mat.use_nodes = True
bsdf = gray_mat.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.45, 0.45, 0.45, 1.0)
    bsdf.inputs['Roughness'].default_value  = 0.85

# ── 复制上半球 → 生成下半球 ─────────────────────────────
bpy.ops.object.select_all(action='DESELECT')
blender_obj.select_set(True)
bpy.context.view_layer.objects.active = blender_obj
bpy.ops.object.duplicate()
bottom_obj = bpy.context.active_object
bottom_obj.name = "bottom_half"

# 把下半球所有材质替换成灰色
bottom_obj.data.materials.clear()
bottom_obj.data.materials.append(gray_mat)

# 翻转所有面法线（镜像后需要翻转才能让法线朝外）
bm = bmesh.new()
bm.from_mesh(bottom_obj.data)
bmesh.ops.reverse_faces(bm, faces=list(bm.faces))

# 将每个顶点的 Z 关于赤道镜像：z_new = 2*z_equator - z_old
for v in bm.verts:
    v.co.z = 2 * z_equator - v.co.z

bm.to_mesh(bottom_obj.data)
bm.free()
bottom_obj.data.update()
print("下半球（镜像）生成完毕")

# ── 合并上下两个对象 ────────────────────────────────────
bpy.ops.object.select_all(action='DESELECT')
blender_obj.select_set(True)
bottom_obj.select_set(True)
bpy.context.view_layer.objects.active = blender_obj
bpy.ops.object.join()
merged_obj = bpy.context.active_object

# ── 焊接赤道接缝顶点 ────────────────────────────────────
print("焊接赤道接缝...")
bm2 = bmesh.new()
bm2.from_mesh(merged_obj.data)
before = len(bm2.verts)
bmesh.ops.remove_doubles(bm2, verts=bm2.verts, dist=0.004)
after = len(bm2.verts)
print(f"  合并顶点: {before} → {after}")
bm2.to_mesh(merged_obj.data)
bm2.free()
merged_obj.data.update()

# ── 整体法线重算 ─────────────────────────────────────────
bm3 = bmesh.new()
bm3.from_mesh(merged_obj.data)
# 计算包围盒中心
bm3.verts.ensure_lookup_table()
bbox_center = Vector([
    (min(v.co[i] for v in bm3.verts) + max(v.co[i] for v in bm3.verts)) / 2
    for i in range(3)
])
# 翻转朝内的面
to_flip = [
    f for f in bm3.faces
    if f.normal.dot(f.calc_center_median() - bbox_center) < 0
]
if to_flip:
    bmesh.ops.reverse_faces(bm3, faces=to_flip)
    print(f"  修正法线: 翻转 {len(to_flip)} 个面")
bm3.to_mesh(merged_obj.data)
bm3.free()
merged_obj.data.update()

# ── 导出 ────────────────────────────────────────────────
print(f"\n导出到: {OUTPUT_OBJ}")
bpy.ops.object.select_all(action='DESELECT')
merged_obj.select_set(True)
bpy.context.view_layer.objects.active = merged_obj
bpy.ops.wm.obj_export(filepath=OUTPUT_OBJ, export_selected_objects=True)
print("Done!")
print(f"  完整球体已保存: {OUTPUT_OBJ}")
print("  上半球：Meshroom 原始纹理")
print("  下半球：灰色材质（镜像几何）")
