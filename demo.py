from ultralytics import YOLO

# 1. 加载模型 (首次运行会自动下载权重文件)
# 'n' 代表 nano (最快)，也可以换成 's', 'm', 'l', 'x' (精度更高但更慢)
model = YOLO('yolo11x.pt') 

# 2. 进行预测
# source=0 代表摄像头，也可以是 'image.jpg' 或 'video.mp4'
# conf=0.5 表示只显示置信度大于 0.5 的结果
# 将分辨率提高到 1280，置信度设为 0.25
results = model.predict(source=0, show=True, imgsz=1280, conf=0.25)

# 代码运行后，按下 'q' 键或关闭窗口即可退出