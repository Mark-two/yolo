import cv2
import os
import glob

# 1. 设置配置
video_path_list = glob.glob("no-cat-ball.mp4") # 自动寻找当前目录下所有的 .mp4 文件
output_folder = "my_data_no_cat_ball"            # 图片输出文件夹
fps_interval = 30                    # 假设视频是30帧/秒，这里设30就是每秒取1张

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)
img_count = 0

print(f"找到视频文件: {video_path_list}")

for video_file in video_path_list:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_file} (可能是路径错误或缺少解码器)")
        continue
    
    print(f"正在处理: {video_file} ...")
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔 fps_interval 帧保存一次
        if frame_id % fps_interval == 0:
            save_path = os.path.join(output_folder, f"ball_{img_count:03d}.jpg")
            cv2.imwrite(save_path, frame)
            # print(f"已保存: {save_path}") # 嫌刷屏可以注释掉
            img_count += 1
        
        frame_id += 1
    
    cap.release()

print(f"\n✅ 全部搞定！共生成 {img_count} 张图片。")
print(f"图片保存在 '{output_folder}' 文件夹中。")