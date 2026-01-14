from ultralytics import YOLO
import os

# 1. 定位您的模型路径
# 注意：这就是您终端里提示的那个路径
model_path = '/home/kang/Documents/yolo/runs/detect/runs/detect/cat_ball3_train2/weights/best.pt'

# 检查一下文件是否真的存在，防止路径写错
if not os.path.exists(model_path):
    print(f"❌ 找不到模型文件: {model_path}")
    exit()

# 2. 加载您的专属模型
model = YOLO(model_path)

print("🚀 模型加载成功！按 'q' 键退出...")

# 3. 开始预测
# conf=0.5: 只有置信度大于 50% 才画框 (可以根据效果调整，太灵敏就调高，检测不到就调低)
results = model.predict(source=0, show=True, conf=0.8)