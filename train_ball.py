from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载模型
    # 我们使用 yolo11n.pt (Nano版) 作为预训练底座
    # 它的参数最少，训练最快，最适合在 Jetson/树莓派上部署
    print("正在加载模型...")
    model = YOLO('yolo11s.pt') 

    # 2. 开始训练 (Fine-tuning)
    # data: 指向数据集的配置文件 (请检查路径是否正确)
    # epochs: 训练轮数 (100轮通常能达到很好的效果，想快点看结果可以改成 50)
    # imgsz: 输入图像大小 (640 是标准)
    # batch: 批次大小 (默认 16，如果显存报错 OOM，可以改小成 8 或 4)
    print("开始训练...")
    results = model.train(
        data='datasets/cat-ball3/data.yaml', 
        epochs=200, 
        imgsz=640,
        project='runs/detect', # 结果保存的根目录
        name='cat_ball3_train',  # 这一波训练任务的名字
        batch = 64
    )

    print(f"训练完成！")
    print(f"最佳模型保存在: runs/detect/cat_ball2_train/weights/best.pt")