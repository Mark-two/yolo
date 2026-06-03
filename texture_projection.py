#!/usr/bin/env python3
"""Texture projection: project real image colors onto synthetic mesh."""
import os, sys
import numpy as np
import torch
import trimesh
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dust3r"))
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

torch.backends.cuda.matmul.allow_tf32 = True

# ── Config ──
REAL_IMG_DIR = "/home/kang/Documents/yolo/my_data"
MESH_PATH = "/home/kang/Documents/yolo/meshroom_synth_output/ball_clean.obj"
OUTPUT_DIR = "/home/kang/Documents/yolo/texture_projection"
MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
DEVICE = "cuda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("Texture Projection Pipeline")
print("="*60)

# ── 1. Get DUSt3R camera poses ──
print("\n[1/5] Estimating camera poses from real images...")
model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_NAME).to(DEVICE)
image_files = sorted([os.path.join(REAL_IMG_DIR, f) for f in os.listdir(REAL_IMG_DIR) if f.endswith('.jpg')])
imgs = load_images(image_files, size=512, verbose=False, patch_size=model.patch_size, square_ok=False)
pairs = make_pairs(imgs, scene_graph='swin-5', prefilter=None, symmetrize=True)
output = inference(pairs, model, DEVICE, batch_size=1, verbose=False)
scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init='mst', niter=200, schedule='cosine', lr=0.01)

cams2world = scene.get_im_poses().detach().cpu().numpy()
focals = scene.get_focals().detach().cpu().numpy()
du_h, du_w = imgs[0]['img'].shape[-2:]  # image size used by DUSt3R
print(f"  DUSt3R image size: {du_w}x{du_h}")
print(f"  Cameras: {len(cams2world)}")

del model, output, scene, imgs, pairs
torch.cuda.empty_cache()

# ── 2. Load synthetic mesh ──
print("\n[2/5] Loading clean synthetic mesh...")
mesh = trimesh.load(MESH_PATH)
if isinstance(mesh, trimesh.Scene):
    mesh = list(mesh.geometry.values())[0]
print(f"  Vertices: {len(mesh.vertices):,}, Faces: {len(mesh.faces):,}")
mesh_center = mesh.vertices.mean(axis=0)
mesh_scale = np.linalg.norm(mesh.vertices - mesh_center, axis=1).max()

# ── 3. Align coordinate systems ──
print("\n[3/5] Aligning coordinate systems...")
cam_centers = cams2world[:, :3, 3]
dust3r_center = cam_centers.mean(axis=0)
dust3r_scale = np.linalg.norm(cam_centers - dust3r_center, axis=1).max()

# Transform: mesh points (in mesh coords) → DUSt3R world space
# mesh_point_world = (mesh_point - mesh_center) * (dust3r_scale / mesh_scale) + dust3r_center
s = dust3r_scale / mesh_scale
print(f"  Scale factor: {s:.4f}")
print(f"  Mesh center: {mesh_center}")
print(f"  DUSt3R center: {dust3r_center}")

mesh_vertices = (mesh.vertices - mesh_center) * s + dust3r_center
mesh_normals = mesh.vertex_normals  # already unit length

# ── 4. Project colors onto mesh ──
print("\n[4/5] Projecting real image colors onto mesh...")

# Initialize vertex color accumulator
vertex_colors = np.zeros((len(mesh_vertices), 3), dtype=np.float64)
vertex_weights = np.zeros(len(mesh_vertices), dtype=np.float64)

# For each camera, project visible vertices and accumulate colors
for cam_idx in range(len(cams2world)):
    c2w = cams2world[cam_idx]  # camera-to-world
    w2c = np.linalg.inv(c2w)
    R_cam = w2c[:3, :3]
    t_cam = w2c[:3, 3]
    
    # Load real image
    img_path = image_files[cam_idx]
    real_img = cv2.imread(img_path)
    if real_img is None:
        continue
    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB).astype(np.float64)
    h_orig, w_orig = real_img.shape[:2]
    
    # Scale factor for DUSt3R resized image vs original
    scale_x = du_w / w_orig
    scale_y = du_h / h_orig
    
    # Camera intrinsic (in DUSt3R image space)
    f = float(focals[cam_idx])
    cx, cy = du_w / 2, du_h / 2
    
    # Transform mesh vertices to camera space
    pts_cam = (R_cam @ mesh_vertices.T).T + t_cam  # (N, 3)
    
    # Project to image
    depth = pts_cam[:, 2]
    valid_depth = depth > 0.01
    
    proj_x = (f * pts_cam[valid_depth, 0] / pts_cam[valid_depth, 2] + cx).astype(np.int32)
    proj_y = (f * pts_cam[valid_depth, 1] / pts_cam[valid_depth, 2] + cy).astype(np.int32)
    
    # Map back to original image coordinates
    proj_x_orig = (proj_x / scale_x).astype(np.int32)
    proj_y_orig = (proj_y / scale_y).astype(np.int32)
    
    valid_pix = (
        (proj_x_orig >= 0) & (proj_x_orig < w_orig) &
        (proj_y_orig >= 0) & (proj_y_orig < h_orig)
    )
    
    # View direction (camera → vertex)
    cam_origin = c2w[:3, 3]
    view_dirs = mesh_vertices - cam_origin
    view_dirs = view_dirs / (np.linalg.norm(view_dirs, axis=1, keepdims=True) + 1e-8)
    
    # Dot product with vertex normals → visibility check
    normal_dot = np.sum(mesh_normals * (-view_dirs), axis=1)
    facing_forward = normal_dot > 0.1
    
    # Combine depth and facing
    valid_indices = np.where(valid_depth)[0]
    valid_pixel_mask = np.zeros(len(mesh_vertices), dtype=bool)
    
    for vi, didx in enumerate(valid_indices):
        if valid_pix[vi] and facing_forward[didx]:
            valid_pixel_mask[didx] = True
    
    valid_v = np.where(valid_pixel_mask)[0]
    print(f"  Cam {cam_idx+1}/{len(cams2world)}: {len(valid_v)} visible verts")
    
    if len(valid_v) == 0:
        continue
    
    # Collect colors from image
    depth_v = pts_cam[valid_v, 2]
    proj_x_v = (f * pts_cam[valid_v, 0] / depth_v + cx).astype(np.int32)
    proj_y_v = (f * pts_cam[valid_v, 1] / depth_v + cy).astype(np.int32)
    proj_x_v_orig = (proj_x_v / scale_x).astype(np.int32)
    proj_y_v_orig = (proj_y_v / scale_y).astype(np.int32)
    
    # Clamp
    proj_x_v_orig = np.clip(proj_x_v_orig, 0, w_orig - 1)
    proj_y_v_orig = np.clip(proj_y_v_orig, 0, h_orig - 1)
    
    colors = real_img[proj_y_v_orig, proj_x_v_orig] / 255.0
    
    # Weight by normal-facing angle
    weights = normal_dot[valid_v]
    
    vertex_colors[valid_v] += colors * weights[:, None]
    vertex_weights[valid_v] += weights

# Normalize
valid_final = vertex_weights > 0
vertex_colors[valid_final] /= vertex_weights[valid_final, None]
print(f"  Final colored vertices: {valid_final.sum():,}/{len(mesh_vertices):,}")

# Fill uncolored vertices with nearest colored vertex color
if valid_final.sum() < len(mesh_vertices):
    print("  Filling missing vertices...")
    from scipy.spatial import cKDTree
    colored_pts = mesh_vertices[valid_final]
    uncolored_pts = mesh_vertices[~valid_final]
    tree = cKDTree(colored_pts)
    _, idx = tree.query(uncolored_pts)
    vertex_colors[~valid_final] = vertex_colors[valid_final][idx]
    vertex_weights[~valid_final] = 1.0

# ── 5. Save vertex-colored mesh ──
print("\n[5/6] Saving vertex-colored mesh...")
new_mesh = trimesh.Trimesh(
    vertices=mesh_vertices,
    faces=mesh.faces,
    vertex_colors=(vertex_colors * 255).astype(np.uint8)
)
vc_path = os.path.join(OUTPUT_DIR, "ball_vertex_color.obj")
new_mesh.export(vc_path)
print(f"  Saved: {vc_path}")

# ── 6. UV unwrap and bake texture ──
print("\n[6/6] UV unwrapping and baking texture...")
import xatlas

vmapping, indices, uvs = xatlas.parametrize(mesh_vertices.astype(np.float64), mesh.faces.astype(np.int32))
print(f"  UV atlas: {len(uvs):,} UVs, {len(indices):,} indices")

# Create texture atlas by rasterizing each triangle
tex_size = 1024
texture = np.zeros((tex_size, tex_size, 3), dtype=np.uint8)
remapped_colors = vertex_colors[vmapping]

for tri_idx in range(0, len(indices), 3):
    i0, i1, i2 = indices[tri_idx], indices[tri_idx+1], indices[tri_idx+2]
    uv0, uv1, uv2 = uvs[i0], uvs[i1], uvs[i2]
    c0, c1, c2 = remapped_colors[i0], remapped_colors[i1], remapped_colors[i2]
    
    u_min = int(min(float(uv0[0]), float(uv1[0]), float(uv2[0])) * tex_size)
    u_max = int(max(float(uv0[0]), float(uv1[0]), float(uv2[0])) * tex_size) + 1
    v_min = int(min(float(1-uv0[1]), float(1-uv1[1]), float(1-uv2[1])) * tex_size)
    v_max = int(max(float(1-uv0[1]), float(1-uv1[1]), float(1-uv2[1])) * tex_size) + 1
    
    u_min = max(0, min(u_min, tex_size))
    u_max = max(0, min(u_max, tex_size))
    v_min = max(0, min(v_min, tex_size))
    v_max = max(0, min(v_max, tex_size))
    
    for v in range(v_min, v_max):
        for u in range(u_min, u_max):
            p = np.array([u / tex_size, 1 - v / tex_size])
            # Barycentric using area
            area = abs((uv1[0]-uv0[0])*(uv2[1]-uv0[1]) - (uv2[0]-uv0[0])*(uv1[1]-uv0[1]))
            if area < 1e-10:
                continue
            w0 = abs((uv1[0]-p[0])*(uv2[1]-p[1]) - (uv2[0]-p[0])*(uv1[1]-p[1])) / area
            w1 = abs((uv2[0]-p[0])*(uv0[1]-p[1]) - (uv0[0]-p[0])*(uv2[1]-p[1])) / area
            w2 = abs((uv0[0]-p[0])*(uv1[1]-p[1]) - (uv1[0]-p[0])*(uv0[1]-p[1])) / area
            
            if w0 + w1 + w2 > 1.001:
                continue
            
            color = (w0 * c0 + w1 * c1 + w2 * c2) * 255
            texture[v, u] = np.clip(color, 0, 255).astype(np.uint8)
    
    if tri_idx % 10000 == 0:
        print(f"  Rasterizing tri {tri_idx}/{len(indices)}")

# Save texture
tex_path = os.path.join(OUTPUT_DIR, "ball_texture.png")
Image.fromarray(texture).save(tex_path)
print(f"  Saved: {tex_path}")

# Export UV-textured mesh
remapped_verts = mesh_vertices[vmapping]
new_mesh_uv = trimesh.Trimesh(
    vertices=remapped_verts,
    faces=indices.reshape(-1, 3),
    visual=trimesh.visual.texture.TextureVisuals(uv=uvs, image=Image.fromarray(texture))
)
obj_path = os.path.join(OUTPUT_DIR, "ball_textured.obj")
new_mesh_uv.export(obj_path)
print(f"  Saved: {obj_path}")

print(f"\n{'='*60}")
print(f"Done! Output:")
print(f"  {obj_path}")
print(f"  {vc_path}")
print(f"  {tex_path}")
print(f"{'='*60}")
