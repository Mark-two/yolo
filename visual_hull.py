#!/usr/bin/env python3
"""Visual Hull reconstruction from SAM masks and DUSt3R camera poses."""
import os
import sys
import numpy as np
import torch
import trimesh
from PIL import Image
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dust3r"))

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

torch.backends.cuda.matmul.allow_tf32 = True

# ── Config ──
MASK_DIR = "/home/kang/Documents/yolo/sam_ball_masked"
INPUT_DIR = "/home/kang/Documents/yolo/my_data"
OUTPUT_DIR = "/home/kang/Documents/yolo/visual_hull_output"
MODEL_NAME = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
DEVICE = "cuda"
GRID_RES = 128  # Voxel grid resolution
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("Visual Hull: Silhouette-based 3D Reconstruction")
print("="*60)

# ── 1. Get camera poses from DUSt3R ──
print("\n[1/3] Estimating camera poses from DUSt3R...")
model = AsymmetricCroCo3DStereo.from_pretrained(MODEL_NAME).to(DEVICE)
image_files = sorted([os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.endswith('.jpg')])
imgs = load_images(image_files, size=512, verbose=False, patch_size=model.patch_size, square_ok=False)
pairs = make_pairs(imgs, scene_graph='swin-5', prefilter=None, symmetrize=True)
output = inference(pairs, model, DEVICE, batch_size=1, verbose=False)
scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PointCloudOptimizer)
scene.compute_global_alignment(init='mst', niter=200, schedule='cosine', lr=0.01)

# Extract camera poses (camera-to-world)
cams2world = scene.get_im_poses().detach().cpu().numpy()  # (N, 4, 4)
focals = scene.get_focals().detach().cpu().numpy()        # (N,)
print(f"  Got poses for {len(cams2world)} cameras")

# Free GPU memory
del model, output, scene
torch.cuda.empty_cache()

# ── 2. Load SAM masks ──
print("\n[2/3] Loading SAM masks...")
mask_files = sorted([os.path.join(MASK_DIR, f) for f in os.listdir(MASK_DIR) if f.endswith('.jpg')])
masks = []
img_h, img_w = None, None
for mf in mask_files:
    # SAM output: black background, non-black = foreground
    img = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    if img_h is None:
        img_h, img_w = img.shape
    else:
        img = cv2.resize(img, (img_w, img_h))
    mask = (img > 10).astype(np.uint8)  # threshold near-black
    # Erode slightly to avoid edge artifacts
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    masks.append(mask)

# Align masks with DUSt3R images (DUSt3R resizes to 512 max dim)
# DUSt3R results: images[0].shape is the image size used
du_size_h, du_size_w = imgs[0]['img'].shape[-2:]  # e.g., 288x512
print(f"  DUSt3R image size: {du_size_w}x{du_size_h}")
print(f"  Original mask size: {img_w}x{img_h}")
print(f"  Loaded {len(masks)} masks")

# Resize masks to DUSt3R resolution
masks_resized = []
for m in masks:
    m_resized = cv2.resize(m, (du_size_w, du_size_h), interpolation=cv2.INTER_NEAREST)
    masks_resized.append(m_resized)
masks = masks_resized

# Build projection matrices P = K [R|t]
# K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
cx, cy = du_size_w / 2, du_size_h / 2

projections = []
for i in range(len(cams2world)):
    c2w = cams2world[i]  # 4x4 camera-to-world
    # world-to-camera = inverse of c2w
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]
    
    # Focal length (DUSt3R output)
    f = float(focals[i])
    
    # Intrinsic matrix
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    
    # Projection matrix P = K [R | t]
    P = K @ np.hstack([R, t.reshape(3, 1)])
    projections.append(P)

print(f"  Built {len(projections)} projection matrices")

# ── 3. Space carving (Visual Hull) ──
print(f"\n[3/3] Space carving (grid {GRID_RES}^3)...")

# Determine bounding box from camera centers
cam_centers = cams2world[:, :3, 3]
bbox_min = cam_centers.min(axis=0) - 0.5
bbox_max = cam_centers.max(axis=0) + 0.5
# Center around the mean position (ball should be at center of camera views)
center = (bbox_min + bbox_max) / 2
extent = (bbox_max - bbox_min).max() / 2
bbox_min = center - extent
bbox_max = center + extent
print(f"  Bounding box: {bbox_min} -> {bbox_max}")

# Create voxel grid
x = np.linspace(bbox_min[0], bbox_max[0], GRID_RES)
y = np.linspace(bbox_min[1], bbox_max[1], GRID_RES)
z = np.linspace(bbox_min[2], bbox_max[2], GRID_RES)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
voxels = np.ones((GRID_RES, GRID_RES, GRID_RES), dtype=np.uint8)

# Space carving: for each camera, project voxels and carve
voxel_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)  # (N^3, 3)
voxel_pts_h = np.hstack([voxel_pts, np.ones((len(voxel_pts), 1))])  # homogeneous

for i in range(len(projections)):
    if masks[i].sum() < 100:  # Skip if mask too small
        continue
    
    # Project all voxels to this camera
    proj = (projections[i] @ voxel_pts_h.T).T  # (N^3, 3)
    depths = proj[:, 2]
    mask_valid_depth = depths > 0.01
    
    proj_x = (proj[:, 0] / depths).astype(np.int32)
    proj_y = (proj[:, 1] / depths).astype(np.int32)
    
    # Check which projected points are inside the mask
    valid_coords = (
        mask_valid_depth &
        (proj_x >= 0) & (proj_x < du_size_w) &
        (proj_y >= 0) & (proj_y < du_size_h)
    )
    
    inside = np.zeros(len(voxel_pts), dtype=bool)
    inside[valid_coords] = masks[i][proj_y[valid_coords], proj_x[valid_coords]] > 0
    
    # Carve: voxels not inside silhouette are removed
    voxels_flat = voxels.ravel()
    voxels_flat[~inside] = 0
    
    if (i+1) % 10 == 0:
        remaining = voxels_flat.sum()
        print(f"  Camera {i+1}/{len(projections)}: {remaining:,} voxels remaining")

print(f"  Final: {voxels.sum():,} voxels")

# ── 4. Extract mesh from voxels ──
print("\n[4/5] Extracting mesh...")
# Use trimesh marching cubes
from skimage import measure
try:
    verts, faces, _, _ = measure.marching_cubes(voxels, level=0.5)
    verts = verts / (GRID_RES - 1) * (bbox_max - bbox_min) + bbox_min
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    
    # Smooth
    mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=3)
    
    # Remove small disconnected components
    components = trimesh.graph.split(mesh, only_watertight=False)
    components = sorted(components, key=lambda x: len(x.faces), reverse=True)
    mesh = components[0]
    
    print(f"  Mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
except Exception as e:
    print(f"  Marching cubes failed: {e}")
    # Fallback: use voxel centroids as point cloud
    v_indices = np.argwhere(voxels > 0)
    v_pts = v_indices.astype(np.float32) / (GRID_RES - 1) * (bbox_max - bbox_min) + bbox_min
    mesh = trimesh.PointCloud(v_pts)
    print(f"  Fallback to point cloud: {len(v_pts):,} points")

# Export
obj_path = os.path.join(OUTPUT_DIR, "visual_hull.obj")
mesh.export(obj_path)
print(f"  Exported: {obj_path}")

print(f"\n{'='*60}")
print(f"Done! Output: {obj_path}")
print(f"{'='*60}")
