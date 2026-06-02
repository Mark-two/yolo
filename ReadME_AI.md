# 图片 → 3D模型 重建方案对比结果

| 方案 | 脚本/输出 | 输入 | 是否完整球 | 底部处理 | 孔洞 | 纹理 | 核心问题 |
|------|----------|------|-----------|---------|------|------|----------|
| 程序化球体 | `gen_ball.py` | 无（Blender建模） | ✅ 完美 | N/A | 无 | 随机色 | 非重建，仅合成基准 |
| Meshroom 真实扫描 | `meshroom_real_output/` | D435i 100+张 | ❌ 半球 | SAM2+RANSAC+Z切，仍有残渣 | 大量 | 模糊 | 光滑面 SIFT 特征不足 |
| Meshroom 合成重建 | `meshroom_synth_output/` | 合成渲染 90张 | ❌ 半球 | RANSAC去底留残渣 | 少量 | 棋盘格污染 | 底部遮挡物理无解 |
| 3DGS splatfacto | `nerf_output/` | 真实图片 | 可渲染 | 无法导出mesh | 无 | ⭐⭐⭐⭐⭐ | 导出不了网格，进不了BlenderProc |
| Visual Hull | `visual_hull_output/` | SAM mask + DUSt3R位姿 | ❌ 粗糙体素 | carving不干净 | 低分辨率128³ | 无 | 精度太低 |
| DUSt3R直接重建 | `dust3r/output*/` | 原图/抠图 | ❌ 稀疏点云 | 底部融地板 | 稀疏 | 点云色 | 非网格 |
| TripoSR | `TripoSR/output/` | 1张图 | ❌ 饼状 | AI推测 | 无 | AI生成 | shape不精准，纹理丢失 |
| **InstantMesh** ⭐ | `InstantMesh/outputs/` | 1张图 | ✅ 完整球体 | AI推测，收敛 | 无 | ⭐⭐⭐ AI纹理 | 纹理非实拍，但形状好 |
| 半球镜像补全 | `complete_sphere.py` | Meshroom结果 | ✅ 完整 | 下半灰色 | 赤道接缝 | 上半有下半灰 | 纹理不真实 |

## 最佳方案：InstantMesh（单图→3D）

经过大量对比实验，**InstantMesh 是目前实拍图重建效果最好的方案**：

| 模型 | 形状 | 纹理 | 备注 |
|------|------|------|------|
| DUSt3R | ❌ 全是地板 | - | 纹理不足，丢失物体 |
| Meshroom | ❌ 特征匹配失败 | - | SIFT 匹配不上 |
| TripoSR | ❌ 饼状 | - | 深度估计不准 |
| **InstantMesh (large)** | ✅ 完整球 | ⭐⭐⭐ | 扩散模型+网格重建 |
| 合成数据+Meshroom | ✅ 完整球 | 程序化花纹 | 非实拍纹理 |

### 安装与运行

```bash
# 1. 克隆仓库
git clone --recursive https://github.com/TencentARC/InstantMesh

# 2. 安装依赖（在 yolo conda 环境）
conda install -n yolo ninja -y
conda run -n yolo pip install "diffusers==0.20.2" "transformers==4.34.1" \
  "pytorch-lightning==2.1.2" "huggingface-hub==0.22.2"
conda run -n yolo pip install git+https://github.com/NVlabs/nvdiffrast/ --no-build-isolation

# 3. 运行（large 模型，grid_res=64 适配 16GB VRAM）
XFORMERS_DISABLED=1 python run.py configs/instant-mesh-large-64.yaml \
  my_data/ball_000.jpg --save_video --export_texmap
```

### 关键技术细节

- **Zero123++ 扩散模型**：单图生成 6 个不同视角的视图
- **LRM 网格重建**：从 6 个视图重建完整 3D mesh
- **FlexiCubes**：自适应等值面提取，质量高于 marching cubes
- **纹理烘焙**：`--export_texmap` 生成 UV 贴图（vs 默认顶点颜色）

### 限制

- 纹理基于 AI 扩散模型生成，非实拍精确颜色
- 需要 ~15GB VRAM（large 模型 grid_res=128 会 OOM，用 grid_res=64 可跑）
- xformers 在 PyTorch 2.9.1 上有兼容问题，用 `XFORMERS_DISABLED=1` 跳过

### 结论

InstantMesh 解决了"球不是球"的问题。纹理虽非实拍，但形状精准，可输入 BlenderProc 渲染合成训练数据。
纹理问题可用实拍投影方案缓解（`texture_projection.py`），但效果不稳定。

---

# 新增方案：RealSense 深度融合 TSDF

## 为什么之前所有方案都失败？

根源：**SfM（COLMAP/Meshroom）依赖图像 SIFT 特征点匹配来估计相机位姿。**  
猫玩具球表面光滑、缺少纹理 → 特征点极少 → COLMAP 只注册了 2/84 帧 → 重建崩溃。  
splatfacto 的 nerf_data/transforms.json 里也只有 2 个相机位姿，训练从未收敛。

## TSDF 深度融合为什么可行？

RealSense D435i 是**主动深度相机**，用红外结构光测量距离，**不依赖物体表面纹理**。  
TSDF（Truncated Signed Distance Function）融合直接从深度图构建体积网格，完全绕过 SfM。

## 管线

```
RealSense D435i 绕拍 → 存 RGB-D 帧 (capture_depth.py)
    │
    ▼
估计圆形轨迹 + ICP 颜色精化 (fuse_depth.py)
    │
    ▼
TSDF 体积融合 → marching cubes 提取网格 → fused_mesh.ply
    │
    ▼
complete_sphere.py 补底 → BlenderProc 渲染 → YOLO 训练
```

## 脚本使用

```bash
# 步骤 1：lekiwi 底盘绕球行驶，每个角度按 S 保存一帧（建议 30-60 帧均匀分布）
conda run -n yolo python capture_depth.py

# 步骤 2：深度融合 + 网格提取
conda run -n yolo python fuse_depth.py

# 步骤 3：结果在 captured_depth/fused_mesh.ply
```

## 环境依赖

yolo conda 环境已安装：
- `pyrealsense2` — RealSense SDK
- `open3d 0.19.0` — TSDF 融合 + ICP
- `cv2` — ArUco 标记检测（预留）

## 已验证

- [x] splatfacto 失败根因确认：nerf_data/transforms.json 仅 2/84 帧注册（2026-06-02）
- [x] capture_depth.py + fuse_depth.py 代码就绪
- [ ] 实际拍摄测试
- [ ] 网格质量评估
- [ ] BlenderProc 渲染 + YOLO 训练
