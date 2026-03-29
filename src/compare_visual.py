import cv2
import os
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def compare(video_path: str, voxel_dir: str):
    """
    同屏对比：左侧原始RGB + YOLO检测，右侧事件Voxel Grid可视化
    """
    video_path = Path(video_path)
    voxel_dir = Path(voxel_dir)

    print(f"🎬 开始同屏对比视频: {video_path.name}")

    # 检查缓存
    if not voxel_dir.exists() or not any(voxel_dir.iterdir()):
        print(f"❌ 错误: Voxel Grid 目录不存在或为空: {voxel_dir}")
        return

    # 加载模型
    print("🤖 加载 YOLO 模型...")
    rgb_model = YOLO('yolo26s.pt')

    # 读取视频和事件帧
    cap = cv2.VideoCapture(str(video_path))
    voxel_files = sorted(
        list(voxel_dir.glob("vis_*.jpg")) +
        list(voxel_dir.glob("vis_*.png"))
    )

    if not voxel_files:
        print("❌ 未找到可视化文件 (vis_*.jpg/png)")
        cap.release()
        return

    print(f"   视频帧数: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"   事件帧数: {len(voxel_files)}")
    print("🎥 按 'Q' 退出, 空格键暂停")

    frame_idx = 0
    paused = False

    while True:
        if not paused:
            ret, rgb_frame = cap.read()
            if not ret:
                print("✅ 视频播放结束")
                break

        # 读取对应的事件帧
        if frame_idx < len(voxel_files):
            event_img = cv2.imread(str(voxel_files[frame_idx]))
            if event_img is None:
                event_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            event_img = np.zeros((720, 1280, 3), dtype=np.uint8)

        # 统一尺寸
        target_height = 720
        rgb_frame = resize_keep_aspect(rgb_frame, target_height)
        event_img = resize_keep_aspect(event_img, target_height)

        # 确保事件图是3通道
        if len(event_img.shape) == 2:
            event_img = cv2.cvtColor(event_img, cv2.COLOR_GRAY2BGR)

        # YOLO检测
        results = rgb_model(rgb_frame, verbose=False)
        rgb_annotated = results[0].plot() if results else rgb_frame

        # 添加标签
        cv2.putText(rgb_annotated, "RGB + YOLO", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(event_img, "Event Camera", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # 添加事件统计信息
        if frame_idx < len(voxel_files):
            # 简单估计事件密度（基于文件名或可以读取实际统计）
            info_text = f"Frame: {frame_idx}/{len(voxel_files)}"
            cv2.putText(event_img, info_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 水平拼接
        combined = cv2.hconcat([rgb_annotated, event_img])

        # 调整窗口大小适应屏幕
        display_scale = 0.8
        display_width = int(combined.shape[1] * display_scale)
        display_height = int(combined.shape[0] * display_scale)
        display_img = cv2.resize(combined, (display_width, display_height))

        cv2.imshow("Event Camera vs RGB (Q:退出 空格:暂停)", display_img)

        key = cv2.waitKey(30) & 0xFF  # ~30fps
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸ 暂停" if paused else "▶ 继续")

        if not paused:
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 对比结束")


def resize_keep_aspect(img, target_height):
    """保持宽高比resize"""
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(img, (new_w, target_height))