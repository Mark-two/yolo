import blenderproc as bproc
import numpy as np
import os
import random
import cv2
import bpy
import bmesh

# -- 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "my_data_meshroom")
IMG_DIR    = os.path.join(OUTPUT_DIR, "images")
LABEL_DIR  = os.path.join(OUTPUT_DIR, "labels")
os.makedirs(IMG_DIR,   exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# -- 参数
NUM_IMAGES         = 200        # 生成图片数量
IMG_W, IMG_H       = 640, 640   # 分辨率
BALL_CLASS         = 0          # YOLO 类别 ID
RANSAC_ITER        = 500        # RANSAC 迭代次数
RANSAC_THRESHOLD   = 0.008      # 平面内点距离阈值（网格单位，视扫描比例调整）
PLANE_DEL_OFFSET   = 0.015      # 额外向上延伸的删除厚度（连带与纸平齐的色块）
MESH_PATH          = os.path.join(os.path.dirname(__file__),
                                  "meshroom_synth_output", "texturedMesh.obj")


# ── RANSAC 平面检测 ────────────────────────────────────────
def ransac_plane(pts, n_iter=500, threshold=0.008):
    """在点云中找顶点数最多的平面（通常就是棋盘格纸）。
    返回 (normal, d)，满足 normal·x + d = 0。"""
    best_count  = 0
    best_plane  = None
    n = len(pts)

    for _ in range(n_iter):
        idx       = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = pts[idx]
        v1, v2    = p2 - p1, p3 - p1
        normal    = np.cross(v1, v2)
        norm_len  = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d       = -np.dot(normal, p1)

        distances = np.abs(pts @ normal + d)
        count     = int(np.sum(distances < threshold))
        if count > best_count:
            best_count = count
            best_plane = (normal.copy(), float(d))

    return best_plane


# ── 连通分量保留最大块 ─────────────────────────────────────
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
    print(f"  连通分量数: {len(components)}, 前5: {sizes[:5]}")
    largest     = max(components, key=len)
    largest_set = {v.index for v in largest}
    to_delete   = [v for v in bm.verts if v.index not in largest_set]
    bmesh.ops.delete(bm, geom=to_delete, context='VERTS')
    print(f"  保留最大分量: {len(largest)} 个顶点")


# ── 主剥离函数 ────────────────────────────────────────────
def extract_object(blender_obj):
    from mathutils import Vector
    mat  = blender_obj.matrix_world
    mesh = blender_obj.data

    pts = np.array([list(mat @ v.co) for v in mesh.vertices], dtype=np.float64)
    print(f"  总顶点数: {len(pts)}, Z 范围: [{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]")

    # 1. RANSAC 找棋盘格平面
    result = ransac_plane(pts, n_iter=RANSAC_ITER, threshold=RANSAC_THRESHOLD)
    if result is None:
        return False
    normal, d = result
    if normal[2] < 0:
        normal, d = -normal, -d
    inliers = int(np.sum(np.abs(pts @ normal + d) < RANSAC_THRESHOLD))
    print(f"  平面法向量: {np.round(normal, 3)}, 内点: {inliers}/{len(pts)}")

    # 2. 删除平面内顶点
    bpy.ops.object.select_all(action='DESELECT')
    blender_obj.select_set(True)
    bpy.context.view_layer.objects.active = blender_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    for v in mesh.vertices:
        wco  = mat @ v.co
        dist = float(np.dot(normal, [wco.x, wco.y, wco.z]) + d)
        v.select = (dist < PLANE_DEL_OFFSET)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    print(f"  删除平面后剩余: {len(mesh.vertices)} 个顶点")

    # 3. 保留最大连通分量
    bm = bmesh.new()
    bm.from_mesh(mesh)
    keep_largest_component(bm)

    # 4. 重心向量逐面修正法线（reverse_faces 修改绕序，真正修复黑色面）
    verts_co = [v.co.copy() for v in bm.verts]
    centroid  = Vector([sum(c[i] for c in verts_co) / len(verts_co) for i in range(3)])
    faces_to_flip = [f for f in bm.faces
                     if f.normal.dot(f.calc_center_median() - centroid) < 0]
    if faces_to_flip:
        bmesh.ops.reverse_faces(bm, faces=faces_to_flip)
    print(f"  翻转法线: {len(faces_to_flip)} 个面")

    # 5. 填充孔洞
    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)
        new_polys = [f for f in bm.faces if len(f.verts) > 3]
        if new_polys:
            bmesh.ops.triangulate(bm, faces=new_polys)
    print(f"  填孔完成，面数: {len(bm.faces)}")

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return True


# ── 1. 初始化 BlenderProc ─────────────────────────────────
bproc.init()

# ── 2. 加载 Meshroom 网格 ─────────────────────────────────
print("Loading Meshroom mesh...")
objs       = bproc.loader.load_obj(MESH_PATH)
target_obj = objs[0]
target_obj.set_cp("category_id", BALL_CLASS + 1)

# ── 3. 自动剥离：RANSAC + 最大连通分量 ────────────────────
print("Extracting object from background...")
ok = extract_object(target_obj.blender_obj)
if not ok:
    raise RuntimeError("Object extraction failed!")

# ── 4. 居中并贴地 ─────────────────────────────────────────
bbox2  = target_obj.get_bound_box()
cx     = (min(p[0] for p in bbox2) + max(p[0] for p in bbox2)) / 2
cy     = (min(p[1] for p in bbox2) + max(p[1] for p in bbox2)) / 2
z_bot  = min(p[2] for p in bbox2)
target_obj.blender_obj.location.x -= cx
target_obj.blender_obj.location.y -= cy
target_obj.blender_obj.location.z -= z_bot

bbox3      = target_obj.get_bound_box()
half_x     = (max(p[0] for p in bbox3) - min(p[0] for p in bbox3)) / 2
half_z     = (max(p[2] for p in bbox3) - min(p[2] for p in bbox3)) / 2
obj_radius = max(half_x, half_z)
print(f"Object radius ~{obj_radius:.4f} m, centered at origin")

# ── 5. 地面 ───────────────────────────────────────────────
floor     = bproc.object.create_primitive("PLANE", size=10)
floor_mat = bproc.material.create("floor_mat")
floor_mat.set_principled_shader_value("Base Color", [0.35, 0.35, 0.35, 1.0])
floor.replace_materials(floor_mat)

# ── 6. 渲染器设置 ─────────────────────────────────────────
bproc.renderer.set_output_format(enable_transparency=False)
bproc.renderer.set_max_amount_of_samples(64)
bproc.camera.set_resolution(IMG_W, IMG_H)

light = bproc.types.Light()
light.set_type("POINT")

# ── 7. 批量渲染 ───────────────────────────────────────────
saved = 0
for i in range(NUM_IMAGES):

    # 随机相机（球面采样）
    cam_dist  = random.uniform(obj_radius * 3, obj_radius * 10)
    cam_theta = random.uniform(0, 2 * np.pi)
    cam_phi   = random.uniform(np.pi / 8, np.pi / 2.5)
    cam_x = cam_dist * np.sin(cam_phi) * np.cos(cam_theta)
    cam_y = cam_dist * np.sin(cam_phi) * np.sin(cam_theta)
    cam_z = cam_dist * np.cos(cam_phi) + obj_radius

    look_at  = [0.0, 0.0, obj_radius]
    cam_pose = bproc.math.build_transformation_mat(
        [cam_x, cam_y, cam_z],
        bproc.camera.rotation_from_forward_vec(
            [look_at[0] - cam_x, look_at[1] - cam_y, look_at[2] - cam_z]
        )
    )
    bproc.camera.add_camera_pose(cam_pose)

    # 随机点光源
    light.set_location([
        random.uniform(-cam_dist, cam_dist),
        random.uniform(-cam_dist, cam_dist),
        random.uniform(obj_radius * 2, obj_radius * 8),
    ])
    light.set_energy(random.uniform(300, 1200))
    light.set_color([1.0, random.uniform(0.85, 1.0), random.uniform(0.75, 1.0)])

    # 渲染 RGB + 分割图
    data     = bproc.renderer.render()
    seg_maps = bproc.renderer.render_segmap(map_by=["instance", "class"])

    class_mask = np.array(seg_maps["class_segmaps"][0])
    obj_pixels  = np.argwhere(class_mask == 1)

    if len(obj_pixels) > 0:
        r_min, c_min = obj_pixels.min(axis=0)
        r_max, c_max = obj_pixels.max(axis=0)

        cx_n = (c_min + c_max) / 2.0 / IMG_W
        cy_n = (r_min + r_max) / 2.0 / IMG_H
        bw_n = (c_max - c_min) / IMG_W
        bh_n = (r_max - r_min) / IMG_H

        img_rgb    = data["colors"][0]
        img_bgr    = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        img_path   = os.path.join(IMG_DIR,   f"mesh_{i:04d}.jpg")
        label_path = os.path.join(LABEL_DIR, f"mesh_{i:04d}.txt")
        cv2.imwrite(img_path, img_bgr)

        with open(label_path, "w") as f:
            f.write(f"{BALL_CLASS} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}\n")

        saved += 1
        print(f"[{i+1}/{NUM_IMAGES}] saved: {img_path}")
    else:
        print(f"[{i+1}/{NUM_IMAGES}] skipped (object not in frame)")

    bproc.utility.reset_keyframes()

print(f"\nDone! {saved} images saved to: {OUTPUT_DIR}")
