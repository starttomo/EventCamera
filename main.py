from src.event_simulator import EventSimulator
from src.preprocess import events_to_voxel_grid
from src.compare_visual import compare
import os
import cv2

if __name__ == "__main__":
    video_name = "road1.mp4"  # 修改为你的视频名

    video_path = f"data/raw/{video_name}"
    events_path = f"data/processed/events/{video_name.replace('.mp4', '.npz')}"
    voxel_dir = f"data/processed/event_frames/{video_name.replace('.mp4', '')}"

    print(f"🚀 正在处理视频: {video_name}")
    print("=" * 50)

    # 检查视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"找不到视频: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"📹 视频信息:")
    print(f"   分辨率: {width} x {height}")
    print(f"   FPS: {fps:.2f}")
    print(f"   总帧数: {total_frames}")
    print(f"   时长: {duration:.2f}s")
    cap.release()

    # 1. 生成事件（如果未生成）
    if not os.path.exists(events_path):
        print("\n⚙️ 步骤1: 生成事件数据...")
        # 关键参数：threshold=0.02（低阈值捕捉更多运动）
        simulator = EventSimulator(threshold=0.02, noise_threshold=0.005)
        events = simulator.simulate(video_path, events_path, fps_boost=4)

        if len(events) == 0:
            print("❌ 错误：未生成任何事件！视频可能无运动或阈值过高")
            exit(1)
    else:
        print(f"\n✅ 步骤1: 事件文件已存在: {events_path}")

    # 2. 生成 Voxel Grid
    print("\n⚙️ 步骤2: 生成 Voxel Grid...")
    # 关键参数：accumulation_time=0.03（高频更新）
    events_to_voxel_grid(
        events_path,
        voxel_dir,
        num_bins=10,  # 更多时间分箱
        accumulation_time=0.03  # 更短累积窗口（33Hz）
    )

    # 3. 同屏对比
    print("\n⚙️ 步骤3: 启动对比可视化...")
    compare(video_path, voxel_dir)