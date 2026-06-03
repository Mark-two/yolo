import blenderproc as bproc
# 只做剥离，不渲染。
# 结果导出到 meshroom_synth_output/extracted.obj，用 Blender 打开检查效果。
import numpy as np
import os
import bpy
import bmesh

RANSAC_ITER      = 500
RANSAC_THRESHOLD = 0.015   # 真实扫描噪声更大，阈值放宽
PLANE_DEL_OFFSET = 0.030   # 切掉底部黑边残渣
MESH_PATH  = os.path.join(os.path.dirname(__file__),
                          "meshroom_real_output", "texturedMesh.obj")
OUTPUT_OBJ = os.path.join(os.path.dirname(__file__),
                          "meshroom_real_output", "extracted.obj")


def ransac_plane(pts, n_iter=500, threshold=0.008):
    best_count = 0
    best_plane = None
    n = len(pts)
    for _ in range(n_iter):
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = pts[idx]
        v1, v2 = p2 - p1, p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)
        distances = np.abs(pts @ normal + d)
        count = int(np.sum(distances < threshold))
        if count > best_count:
            best_count = count
            best_plane = (normal.copy(), float(d))
    return best_plane


def keep_largest_component(bm):
    bm.verts.ensure_lookup_table()
    visited    = [False] * len(bm.verts)
    components = []
    for start in bm.verts:
        if visited[start.index]:
            continue
        comp  = []
        stack = [start]
        while stack:
            v = stack.pop()
            if visited[v.index]:
                continue
            visited[v.index] = True
            comp.append(v)
            for e in v.link_edges:
                other = e.other_vert(v)
                if not visited[other.index]:
                    stack.append(other)
        components.append(comp)

    sizes = sorted([len(c) for c in components], reverse=True)
    print(f"  连通分量数: {len(components)}")
    print(f"  各分量顶点数（前5）: {sizes[:5]}")

    largest     = max(components, key=len)
    largest_set = {v.index for v in largest}
    to_delete   = [v for v in bm.verts if v.index not in largest_set]
    bmesh.ops.delete(bm, geom=to_delete, context='VERTS')
    print(f"  保留最大分量: {len(largest)} 个顶点")


# ── 初始化 ────────────────────────────────────────────────
bproc.init()

print("Loading mesh...")
objs       = bproc.loader.load_obj(MESH_PATH)
target_obj = objs[0]
blender_obj = target_obj.blender_obj

mat  = blender_obj.matrix_world
mesh = blender_obj.data
pts  = np.array([list(mat @ v.co) for v in mesh.vertices], dtype=np.float64)

print(f"总顶点数: {len(pts)}")
print(f"X 范围: [{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]")
print(f"Y 范围: [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]")
print(f"Z 范围: [{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]")

# ── SAM 预处理后的扫描：跳过 RANSAC（地面已被 SAM 删除，无法检测平面）──
print("\n[INFO] SAM 模式：跳过 RANSAC，直接用 Z 最低值切除底部残渣\n")

# ── 直接用 Z_min 切底（SAM 已删除地面，RANSAC 无法找到地面平面）──
# 找到底部最低簇的 Z 值（取最低 2% 分位数，比纯 Z_min 更稳健）
z_min_pct = float(np.percentile(pts[:, 2], 2))
z_cut     = z_min_pct + PLANE_DEL_OFFSET
print(f"Z_min(2%分位): {z_min_pct:.4f}")
print(f"切割线  Z   : {z_cut:.4f}  (保留 Z > 此值的顶点)")

bpy.ops.object.select_all(action='DESELECT')
blender_obj.select_set(True)
bpy.context.view_layer.objects.active = blender_obj
bpy.ops.object.mode_set(mode='OBJECT')

for v in mesh.vertices:
    wco = mat @ v.co
    v.select = (wco.z <= z_cut)               # 纯 Z 轴切割，不受平面倾斜影响

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='VERT')
bpy.ops.mesh.delete(type='VERT')
bpy.ops.object.mode_set(mode='OBJECT')
print(f"\n删除底部顶点后剩余: {len(mesh.vertices)} 个顶点")

# ── 保留最大连通分量 ──────────────────────────────────────
print("\n分析连通分量...")
bm = bmesh.new()
bm.from_mesh(mesh)
keep_largest_component(bm)

print("修复网格...")
# Step 1: 合并重复顶点（修复非流形几何，为法线传播打基础）
verts_before = len(bm.verts)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.002)
print(f"  合并重复顶点: {verts_before} → {len(bm.verts)} 个")

# ── Step 2: 填充孔洞
bm.edges.ensure_lookup_table()
boundary_edges = [e for e in bm.edges if e.is_boundary]
print(f"  边界边数量: {len(boundary_edges)}")
if boundary_edges:
    bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)
    new_polys = [f for f in bm.faces if len(f.verts) > 3]
    if new_polys:
        bmesh.ops.triangulate(bm, faces=new_polys)
    print(f"  填孔完成，面数: {len(bm.faces)}")

# ── Step 3: 全局一致法线传播（处理大面积一致区域）
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
print("  全局法线重算完成")

# ── Step 4: 用包围盒中心校验整体方向，错了就全部翻转
from mathutils import Vector
bm.verts.ensure_lookup_table()
bbox_min = Vector([min(v.co[i] for v in bm.verts) for i in range(3)])
bbox_max = Vector([max(v.co[i] for v in bm.verts) for i in range(3)])
center   = (bbox_min + bbox_max) / 2

outward_count = sum(
    1 for f in bm.faces
    if f.normal.dot(f.calc_center_median() - center) > 0
)
print(f"  法线朝外(recalc后): {outward_count}/{len(bm.faces)} 个面")
if outward_count < len(bm.faces) / 2:
    bmesh.ops.reverse_faces(bm, faces=list(bm.faces))
    print("  整体方向错误，已全部翻转")

# ── Step 4b: 逐面修正残余反面（recalc对孤立patch传播失败时的兜底）
faces_to_flip = [
    f for f in bm.faces
    if f.normal.dot(f.calc_center_median() - center) < 0
]
if faces_to_flip:
    bmesh.ops.reverse_faces(bm, faces=faces_to_flip)
print(f"  逐面修正: 翻转了 {len(faces_to_flip)} 个残余反面")

bm.to_mesh(mesh)
bm.free()
mesh.update()

# ── Step 5: 修复后再做一次最大连通分量（去掉随法线修复产生的孤立碎片）
print("修复后二次连通分量清理...")
bm2 = bmesh.new()
bm2.from_mesh(mesh)
keep_largest_component(bm2)
bm2.to_mesh(mesh)
bm2.free()
mesh.update()
print(f"最终剩余: {len(mesh.vertices)} 个顶点")

# ── 封底：扇形封底法（对任意不规则开口有效）────────────────
print("\n封底：填充底部开口...")
from mathutils import Vector
bm3 = bmesh.new()
bm3.from_mesh(mesh)
bm3.edges.ensure_lookup_table()
bm3.verts.ensure_lookup_table()

boundary = [e for e in bm3.edges if e.is_boundary]
print(f"  底部开口边界边数: {len(boundary)}")

cap_face_count = 0
if boundary:
    # 收集边界顶点，计算平均中心点
    bverts = set()
    for e in boundary:
        bverts.add(e.verts[0])
        bverts.add(e.verts[1])
    center_co = Vector()
    for v in bverts:
        center_co += v.co
    center_co /= len(bverts)
    center_co.z = sum(v.co.z for v in bverts) / len(bverts)  # 压平底面
    center_v = bm3.verts.new(center_co)
    bm3.verts.ensure_lookup_table()
    # 每条边界边 + 中心点 = 一个三角形
    for e in boundary:
        v1, v2 = e.verts[0], e.verts[1]
        try:
            f = bm3.faces.new([center_v, v1, v2])
            cap_face_count += 1
        except Exception:
            pass
    # 让封底面法线朝下
    bm3.faces.ensure_lookup_table()
    for f in bm3.faces:
        if f.is_valid and center_v in f.verts and f.normal.z > 0:
            f.normal_flip()
    print(f"  封底三角形数: {cap_face_count}")
else:
    print("  未找到开口，无需封底")

bm3.to_mesh(mesh)
bm3.free()
mesh.update()

# 给封底面单独加灰色材质
gray_mat = bpy.data.materials.new(name="cap_gray")
gray_mat.use_nodes = True
bsdf = gray_mat.node_tree.nodes.get('Principled BSDF')
if bsdf:
    bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs['Roughness'].default_value  = 0.8
blender_obj.data.materials.append(gray_mat)
gray_idx = len(blender_obj.data.materials) - 1

# 把 Z 最低的那批面（含封底）设为灰色材质
bm4 = bmesh.new()
bm4.from_mesh(mesh)
bm4.faces.ensure_lookup_table()
z_vals = [f.calc_center_median().z for f in bm4.faces]
if z_vals:
    z_min    = min(z_vals)
    z_thresh = z_min + (float(np.percentile(pts[:, 2], 98)) - z_min) * 0.08
    for f in bm4.faces:
        if f.calc_center_median().z <= z_thresh:
            f.material_index = gray_idx
bm4.to_mesh(mesh)
bm4.free()
mesh.update()
print("  灰色材质已设置")

# ── 导出结果 ──────────────────────────────────────────────
print(f"\n导出到: {OUTPUT_OBJ}")
bpy.ops.object.select_all(action='DESELECT')
blender_obj.select_set(True)
bpy.context.view_layer.objects.active = blender_obj
bpy.ops.wm.obj_export(filepath=OUTPUT_OBJ, export_selected_objects=True)
print("Done! 用 Blender 打开 meshroom_real_output/extracted.obj 检查效果。")
