# 扫描 → 3D 重建 → 虚拟训练 识别系统规划

> 目标：用真实物体的照片/扫描数据，重建精准的 3D 模型，
> 再批量渲染合成训练图，配合少量真实图，训练出可在机器人上部署的检测模型。

---

## 整体流程总览

```
真实物体
  │
  ▼
【第一阶段】拍摄 & 扫描          →  原始多视角照片
  │
  ▼
【第二阶段】前景分割（SAM2）     →  干净的物体图（去背景）
  │
  ├──→ 支路A：SfM/MVS 传统重建（Meshroom）→ 纹理三角网格 .obj
  │           ⚠️ 已知问题：光滑物体缺洞、底部遮挡无解
  │
  ├──→ 支路B：3DGS 神经重建（splatfacto）→ 高斯点云，直接渲染★推荐
  │           无需导出网格，缺洞问题天然消失
  │
  └──→ 支路C：单/少图 AI 重建（InstantMesh/TripoSG）→ 完整网格含底面★未来
              4-6 张图直接出完整 3D，AI 自动补全遮挡区域
  │
  ▼
【第三阶段】模型清理 & 纹理（仅支路A需要）→  干净可渲染的 .obj/.glb
  │
  ▼
【第四阶段】合成数据生成（BlenderProc）→  images/ + labels/ (YOLO格式)
  │          相机仰角限制 10°–70°，永不渲染底部，规避遮挡问题
  ▼
【第五阶段】模型训练（YOLO）        →  .pt 权重
  │          合成预训练 → 真实数据微调（两阶段效果最佳）
  ▼
【第六阶段】部署验证（机器人/相机）
```

---

## 第一阶段：拍摄 & 扫描

### 方案对比

| 方案 | 工具 | 优点 | 缺点 | 推荐场景 |
|------|------|------|------|----------|
| 手持绕拍 | 手机/相机 | 零门槛 | 光照不稳定，姿态估计差 | 快速验证 |
| 转台拍摄 | 相机 + 转台 | 光照稳定，姿态精确 | 需要手动/电动转台 | 中小物体 |
| 机器臂拍摄 | RealSense + 机器臂 | 全自动，可重复 | 设备成本高 | 生产流程 |
| 结构光/ToF | RealSense D435i | 直出点云，无需SfM | 近距离噪声大 | 大物体/室内 |
| 激光扫描 | Faro / Artec | 毫米级精度 | 昂贵 | 工业级 |

### 本项目当前方案
- [x] **RealSense D435i** 机器臂绕拍，约 100 张
- [x] 拍摄脚本：`capture.py` / `realsense.py`
- [ ] 待改进：加转台同步触发，让相机姿态更均匀分布

### 拍摄要点
- 建议角度：俯仰 0°/30°/60°，每层 12 张，共 36-60 张起步
- 重叠度 ≥ 60%（相邻帧共视面积）
- 避免纯色/反光/透明物体（SfM 特征点少）
- 背景尽量简单或贴棋盘格纸（后续 RANSAC 去底面）
- 光照：柔光箱 + 均匀漫射，避免镜面高光

---

## 第二阶段：前景分割

### 工具：SAM2（Segment Anything Model 2）

```bash
python sam_segment.py --input datasets/new_captures/ \
                      --output sam_masked/ \
                      --model weights/sam2.1_b.pt
```

- [x] 以画面中心点为 prompt，自动分割主体物体
- [x] 背景涂黑（Meshroom 无法对纯黑提特征点 → 自然忽略背景）
- [ ] 优化：多点 prompt（四周采负样本点）提升边缘精度
- [ ] 优化：视频模式帧间传播，减少逐帧推理时间

### 模型规格

| 模型 | 大小 | 速度 | 精度 |
|------|------|------|------|
| sam2.1_t.pt | ~38MB | ★★★★★ | ★★★ |
| sam2.1_s.pt | ~46MB | ★★★★ | ★★★★ |
| sam2.1_b.pt | ~80MB | ★★★ | ★★★★ | ← 当前
| sam2.1_l.pt | ~224MB | ★★ | ★★★★★ |

---

## 第三阶段A：传统 3D 重建（SfM + MVS）

### 工具：Meshroom（AliceVision）

```bash
# 完整管线（自动）
meshroom_batch --input sam_masked/ --output meshroom_real_output/

# 或通过 SAM 脚本末尾自动调用
python sam_segment.py --meshroom
```

**管线节点：**
1. `CameraInit` → 读取图片 & EXIF 相机参数（焦距由 `add_exif.py` 写入）
2. `FeatureExtraction` → SIFT/AKAZE 关键点检测
3. `ImageMatching` + `FeatureMatching` → 特征匹配，建立共视图
4. `StructureFromMotion` → 估计相机位姿 + 稀疏点云
5. `PrepareDenseScene` → 准备密集重建
6. `DepthMap` + `DepthMapFilter` → 每张图估深度图
7. `Meshing` → 从深度图融合生成网格
8. `MeshFiltering` + `Texturing` → 过滤 & 贴纹理

- [x] 完整管线已跑通：`meshroom_real_output/texturedMesh.obj`
- [x] EXIF 注入脚本：`add_exif.py`（焦距 27mm）
- [ ] 待优化：物体特征少时（光滑球面）启用 `OpenMVG` 或密集 patch 策略

### ⚠️ Meshroom 已知缺陷（实测踩坑）

| 问题 | 根本原因 | 解法 |
|------|----------|------|
| 网格有缺洞 | 光滑/无纹理表面 SIFT 特征点不足，深度估计崩溃 | 换用 3DGS 直接渲染，无需完整网格 |
| 底部永远缺失 | 物理遮挡，相机从未拍到 | ① 渲染时限制仰角不拍底部；② 手动在 Blender 加平面盖底 |
| 抠图后效果变差 | SAM2 去背后共视图变稀疏，SfM 更难收敛 | 背景用纯色代替纯黑，或保留部分背景特征点 |
| 重建耗时长 | MVS 深度图逐帧计算 | 考虑直接用 3DGS（10min vs 数小时）|

> **结论**：Meshroom 适合纹理丰富的大场景重建。对于小物体（尤其光滑、对称），
> **优先使用支路B（3DGS直接渲染）**，网格重建仅作为备选。

---

## 第三阶段B：神经 3D 重建（NeRF / 3D Gaussian Splatting）

### 工具：Nerfstudio

```bash
# 第一步：COLMAP 预处理（位姿估计）
./nerf_process.sh

# 第二步：训练
./nerf_train.sh nerfacto        # 经典 NeRF，质量稳定
./nerf_train.sh splatfacto      # 3DGS，渲染更快

# 第三步：导出网格
./nerf_export.sh <实验名>

# 查看实时训练进度
# 浏览器打开 http://localhost:7007
```

### 方案对比

| 方法 | 训练时间 | 渲染质量 | 可导出网格 | 适合场景 |
|------|----------|----------|-----------|----------|
| nerfacto | ~30min | ★★★★ | 是（marching cubes）| 通用 |
| splatfacto (3DGS) | ~10min | ★★★★★ | 部分（需后处理）| 视觉效果优先 |
| instant-ngp | ~2min | ★★★ | 是 | 快速预览 |

- [x] Nerfstudio 管线：`nerf_process.sh` / `nerf_train.sh` / `nerf_export.sh`
- [ ] 待完成：3DGS 导出网格后烘焙纹理（目前导出 .ply 点云）
- [ ] 待尝试：**Gaussian Opacity Fields** (GOF) 直接从 3DGS 提取干净网格

### ★ 短期推荐：3DGS 直接渲染（跳过网格导出）

```bash
# 训练 3DGS
./nerf_train.sh splatfacto

# 不导出网格，直接在 Nerfstudio viewer 中渲染任意视角
# 或：用 gsplat / threestudio 导出渲染图作为合成训练集
```

优势：
- 渲染质量比 Meshroom 网格更高（无洞、无拓扑错误）
- 训练只需 10 分钟
- 直接渲染输出图片喂给 YOLO，完全绕过网格清理步骤

限制：
- 渲染速度比 BlenderProc 慢（约 1-2s/帧）
- 需要额外工具自动化标注 bbox（暂无现成脚本）

---

## 第四阶段：模型清理 & 预处理（仅 Meshroom 支路需要）

### 工具：Blender + BlenderProc，脚本：`extract_only.py` / `render.py`

| 步骤 | 操作 | 工具 |
|------|------|------|
| 去底面 | RANSAC 检测平面 → 删除底面顶点 | `extract_only.py` |
| 去噪 | 删除孤立小网格 (保留最大连通分量) | Blender BMesh |
| 填孔 | 检测边界边 → 三角剖分填孔 | Blender BMesh |
| 法线重算 | 统一法线朝外 | Blender |
| 纹理检查 | 确认 UV 展开 & 贴图路径正确 | 手动/Blender |

- [x] RANSAC 去底面：`extract_only.py`
- [x] 保留最大连通分量 + 填孔
- [ ] 待改进：支持多个物体实例（当前只处理单个网格）

### 底部缺失的实用处理策略

```
策略1（推荐）：渲染时限制相机仰角 10°–70°，不渲染底部视角
             → 底部缺失对训练完全无影响

策略2：在 Blender 中手动为网格底部加一个平面盖住
      → 修复成本低，适合底面是简单平面的物体

策略3：转台拍摄 + 翻转二次扫描 → ICP 对齐合并
      → 精度最高，但流程复杂，仅工业级需要
```

---

## 第五阶段：合成数据生成（BlenderProc）

### 当前脚本

| 脚本 | 说明 | 输出 |
|------|------|------|
| `gen_ball.py` | 用程序化几何球体渲染，颜色随机 | `my_data_synth/` |
| `gen_meshroom.py` | 加载 Meshroom 重建的真实网格渲染 | `my_data_meshroom/` |
| `render.py` | 含 RANSAC 去底面的完整渲染管线 | `my_data_meshroom/` |

### 合成数据策略

```
多样化维度（每张图随机化）：
  ├── 相机位姿：球面均匀采样（仰角 10°–70°，方位角 0°–360°）
  ├── 光照：随机点光源位置 + 颜色温度（暖/冷）+ 环境光强度
  ├── 背景：随机纯色 / 随机贴图 / CC0 HDRI 环境贴图
  ├── 物体位置：桌面随机偏移
  ├── 物体旋转：随机绕 Z 轴旋转（球对称可忽略）
  └── 相机参数：随机轻微焦距抖动（≤5%）
```

- [x] 球面相机采样
- [x] 随机颜色/材质
- [x] 自动 YOLO 格式标注（bbox 从投影计算）
- [ ] 待加：随机背景贴图（提升域迁移能力）
- [ ] 待加：物体遮挡增强（随机放置干扰物体）
- [ ] 待加：运动模糊 + 镜头畸变（匹配真实相机）

### 数据量参考

| 规模 | 合成图数 | 真实图数 | 目标 mAP50 |
|------|----------|----------|-----------|
| 快速验证 | 200 | 50 | ~0.7 |
| 标准 | 1000 | 200 | ~0.85 |
| 高精度 | 5000 | 500 | ~0.92 |

---

## 第六阶段：YOLO 训练

### 当前方案：YOLOv11

```bash
python train_ball.py
# 或直接调用
yolo detect train data=datasets/combined/data.yaml model=yolo11s.pt epochs=100
```

### 数据集组合策略

```
datasets/
  ├── cat-ball/          # 原始真实数据（Roboflow 标注）
  ├── cat-ball2/         # 补充真实数据
  ├── cat-ball3/         # 第三批真实数据
  ├── my_data_synth/     # gen_ball.py 纯合成（程序化球）
  ├── my_data_meshroom/  # render.py 网格渲染合成
  └── combined/          # ← 混合后的最终训练集
        data.yaml
```

**混合比例建议（domain randomization）：**
- 合成：真实 ≈ 3:1 ~ 5:1（合成数量多，真实质量高）
- 使用 `mosaic=1.0` + `mixup=0.1` 增强

- [x] 训练脚本：`train_ball.py`
- [x] 测试脚本：`test_model.py`
- [x] 可用权重：`yolo11n/s/x.pt`
- [ ] 待尝试：YOLOv11-seg（实例分割，精度更高）
- [ ] 待尝试：Fine-tune on 真实数据（先合成预训练，再真实微调）

---

## 第七阶段：部署验证

### 本地验证

```bash
python demo.py              # 摄像头实时推理
python test_model.py        # 对测试集评估 mAP
```

### 机器人部署

```bash
python capture.py           # 机器臂拍摄 → 自动推理
```

- [ ] 待测：在机器臂末端相机上跑推理，评估实际抓取成功率
- [ ] 待加：6D 位姿估计（在检测基础上估计物体姿态，用于抓取规划）
  - 方案：FoundPose / GDR-Net / FoundationPose（近年主流）

---

## 主流前沿方向参考（2025）

### 3D 重建

| 技术 | 代表工具 | 特点 |
|------|----------|------|
| 3D Gaussian Splatting | Nerfstudio splatfacto | 实时渲染，训练快 |
| 2D Gaussian Splatting | 2DGS | 更易提取干净网格 |
| Gaussian Opacity Fields | GOF | 3DGS + 准确网格提取 |
| Feed-forward 单/少视图重建 | **InstantMesh**, Zero123++, One-2-3-45 | 1-5张图直接出3D |
| 大模型驱动重建 | **TripoSG**, **Stable3D** | 文字/单图 → 3D |

### 合成到真实迁移（Sim-to-Real）

| 策略 | 说明 |
|------|------|
| Domain Randomization | 随机化光照/纹理/背景，覆盖真实分布 |
| Domain Adaptation | 用少量真实数据对合成预训练模型微调 |
| **Physically-based Rendering** | PBR材质 + HDRI，缩小渲染gap |
| Feature-level Alignment | 对抗训练对齐合成/真实特征分布 |

### 识别 + 位姿估计一体化

| 方法 | 说明 | 是否需要 3D 模型 |
|------|------|-----------------|
| FoundationPose | 基础模型，zero-shot 位姿估计 | 是 |
| BundleSDF | 在线同时重建+位姿跟踪 | 否（在线建） |
| GigaPose | 超快速模板匹配位姿 | 是（渲染模板）|
| SAM-6D | SAM + 6D 位姿 | 是 |

---

## 当前进度总览

### 已完成 ✅

- [x] RealSense 拍摄脚本
- [x] SAM2 自动分割管线（`sam_segment.py`）
- [x] Meshroom 完整重建（`meshroom_real_output/`）
- [x] Nerfstudio 训练/导出管线（`nerf_*.sh`）
- [x] RANSAC 去底面 + 填孔（`extract_only.py`）
- [x] BlenderProc 合成渲染（`gen_ball.py` / `gen_meshroom.py` / `render.py`）
- [x] YOLO11 训练 + 推理（`train_ball.py` / `test_model.py` / `demo.py`）
- [x] 多数据集合并配置（`datasets/combined/`）

### 近期待做 🔧（基于实测问题）

- [ ] **render.py 限制仰角 10°–70°**，彻底规避底部缺失问题
- [ ] **随机背景贴图** 接入渲染管线（提升 sim-to-real，当前纯色背景gap大）
- [ ] **真实数据微调流程**：先用合成数据预训练，再用少量真实数据 fine-tune
- [ ] SAM2 多点 prompt（四周加负样本点）提升边缘分割精度
- [ ] 尝试 `splatfacto` 直接渲染合成图，对比 Meshroom 网格渲染效果

### 中期目标 🎯

- [ ] **InstantMesh / TripoSG** 替换 Meshroom（4-6张图直接出完整网格，含底面）
- [ ] 6D 位姿估计集成（FoundationPose 或 GigaPose）
- [ ] 支持多物体类别（扩展到其他猫玩具 / 物品）
- [ ] 端到端管线脚本（一键：拍摄 → 重建 → 渲染 → 训练 → 评估）
- [ ] 机器臂抓取验证（结合位姿估计）

### 长期目标 🚀

- [ ] 接入 **NVIDIA Omniverse Replicator**（PBR物理渲染，sim-to-real gap更小）
- [ ] 接入 **BundleSDF** 在线重建+跟踪（无需离线建模，边用边建）
- [ ] 构建物体数据库（多物体类别，统一管线）

---

## 路线成熟度与工业现状（2025）

### 整体路线在工业界成熟吗？

**成熟，但工业界用的工具和你踩到的坑的解法不一样。**

| 场景 | 工业界主流路线 | 与本项目的差距 |
|------|--------------|---------------|
| 小批量物体（<100种）| 结构光扫描（Shining3D/Artec）→ Blender/Omniverse 渲染 → YOLO | 扫描仪精度远高于 Meshroom，无缺洞问题 |
| 大批量（电商/仓储）| 单图重建（TripoSG）→ Omniverse Replicator 渲染 → DINO/YOLO | 完全跳过多视角拍摄，AI补全遮挡 |
| 机器人抓取 | FoundationPose / BundleSDF 直接做 6D 位姿 | 跳过识别+3D两步，在线估计位姿 |

### Meshroom 的行业定位

Meshroom (SfM+MVS) 最初设计目标是**大场景三维重建**（建筑、地形），而非小物体。
用它做小物体的核心问题：
- 光滑/无纹理表面 → SIFT 特征点不足 → 深度图崩溃 → 缺洞
- 底部永远被桌面遮挡 → 物理上无解
- 2024 年后工业界小物体重建已基本迁移到 **3DGS + 单图AI重建**
