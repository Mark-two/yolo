import cv2
import os
import time

# ================= 配置区 =================
# 图片保存位置
SAVE_FOLDER = "datasets/new_captures" 
# =========================================

# 1. 创建文件夹
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 2. 打开摄像头 (0 通常是默认摄像头)
cap = cv2.VideoCapture(0)

# 设置分辨率 (可选，设为 1280x720 会更清晰，取决于摄像头支持)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ 无法打开摄像头！")
    exit()

print(f"✅ 摄像头已启动！图像将保存在: {SAVE_FOLDER}")
print("-------------------------------------------------")
print("👉 按键指南:")
print("   [S] 键: 拍照 (保存当前画面)")
print("   [Q] 键: 退出程序")
print("-------------------------------------------------")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收画面 (stream end?). Exiting ...")
        break

    # 在画面上显示提示信息
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Saved: {count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Capture (Press S to save, Q to quit)', display_frame)

    # 3. 监听按键
    key = cv2.waitKey(1) & 0xFF

    # 按 's' 保存
    if key == ord('s'):
        # 使用时间戳命名，防止覆盖
        timestamp = int(time.time() * 1000)
        filename = f"{SAVE_FOLDER}/img_{timestamp}.jpg"
        
        cv2.imwrite(filename, frame)
        print(f"📸 已保存: {filename}")
        count += 1
        
        # 视觉反馈：闪一下白屏
        cv2.imshow('Camera Capture (Press S to save, Q to quit)', 
                   cv2.addWeighted(frame, 0.5, 255, 0.5, 0))
        cv2.waitKey(50) 

    # 按 'q' 退出
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 程序结束")