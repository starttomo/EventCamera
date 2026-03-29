import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import torch


def events_to_voxel_grid(events_npz: str, output_dir: str, num_bins=8,
                         accumulation_time=0.04, device='cuda', gpu_batch_size=1000000):
    """
    CPU/GPU混合模式 - 大数据量不爆显存，小数据量用GPU加速
    """
    os.makedirs(output_dir, exist_ok=True)

    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    # 加载事件到CPU内存（不是GPU！）
    print(f"📊 加载事件到CPU内存...")
    data = np.load(events_npz)
    events_np = data['events']
    total_events = len(events_np)
    print(f"   总事件数: {total_events:,}")

    # 获取分辨率
    video_path = str(Path(events_npz).parent.parent / "raw" / Path(events_npz).stem.replace('.npz', '.mp4'))
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    else:
        img_width = int(events_np[:, 1].max()) + 10
        img_height = int(events_np[:, 2].max()) + 10

    print(f"🎬 分辨率: {img_width} x {img_height}")

    # 裁剪坐标
    events_np[:, 1] = np.clip(events_np[:, 1], 0, img_width - 1)
    events_np[:, 2] = np.clip(events_np[:, 2], 0, img_height - 1)

    max_t = events_np[:, 0].max()
    frame_count = int(max_t / accumulation_time) + 1
    print(f"🎯 总帧数: {frame_count} | 分箱: {num_bins} | 设备: {device.upper()}")

    # 预计算帧索引（CPU）
    print("⚡ 预计算时间索引...")
    frame_indices = (events_np[:, 0] / accumulation_time).astype(np.int32)
    bin_indices = ((events_np[:, 0] % accumulation_time) / accumulation_time * num_bins).astype(np.int32)
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # 分离正负事件（CPU）
    pos_mask = events_np[:, 3] == 1
    pos_events = events_np[pos_mask][:, :3]  # 只保留t,x,y
    pos_frames = frame_indices[pos_mask]
    pos_bins = bin_indices[pos_mask]

    neg_mask = events_np[:, 3] == -1
    neg_events = events_np[neg_mask][:, :3]
    neg_frames = frame_indices[neg_mask]
    neg_bins = bin_indices[neg_mask] + num_bins  # 偏移

    # 清理大数组
    del events_np, frame_indices, bin_indices
    import gc
    gc.collect()

    # 预加载视频帧
    print("📥 预加载视频帧...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    original_frames = {}

    for i in range(frame_count):
        target = int(i * accumulation_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret:
            original_frames[i] = frame
    cap.release()
    print(f"   加载了 {len(original_frames)} 个原帧")

    # ========== 逐帧处理（每帧单独GPU计算，显存占用极小）==========
    print(f"🎨 逐帧生成（每帧独立GPU处理）...")

    for frame_idx in tqdm(range(frame_count), desc="生成图像"):
        # 为当前帧创建空体素（CPU上）
        voxel = np.zeros((2 * num_bins, img_height, img_width), dtype=np.float32)

        # 筛选当前帧的事件（CPU）
        pos_in_frame = (pos_frames == frame_idx)
        neg_in_frame = (neg_frames == frame_idx)

        # 如果有事件，用GPU加速scatter（分批）
        if pos_in_frame.any():
            batch_to_gpu_scatter(
                pos_events[pos_in_frame],
                pos_bins[pos_in_frame],
                voxel,
                num_bins,
                img_height,
                img_width,
                gpu_batch_size
            )

        if neg_in_frame.any():
            batch_to_gpu_scatter(
                neg_events[neg_in_frame],
                neg_bins[neg_in_frame],
                voxel,
                num_bins,
                img_height,
                img_width,
                gpu_batch_size
            )

        # 可视化（CPU，很快）
        img = voxel_to_image(voxel, num_bins)

        # 融合原帧
        if frame_idx in original_frames:
            img = blend_with_frame(img, original_frames[frame_idx])

        cv2.imwrite(f"{output_dir}/vis_{frame_idx:06d}.jpg", img)

    print(f"✅ 完成！保存在: {output_dir}")


def batch_to_gpu_scatter(events_xyt, bins, voxel_out, num_bins, height, width, batch_size):
    """
    分批GPU scatter_add，控制显存占用
    """
    total = len(events_xyt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        # 小批量移到GPU
        batch_txy = torch.from_numpy(events_xyt[start:end]).float().to(device)
        batch_bins = torch.from_numpy(bins[start:end]).long().to(device)

        # 坐标转整数
        batch_y = batch_txy[:, 2].long()
        batch_x = batch_txy[:, 1].long()

        # 计算线性索引
        indices = (batch_bins * height * width + batch_y * width + batch_x)

        # scatter_add（GPU）
        voxel_flat = torch.from_numpy(voxel_out).view(-1).float().to(device)
        ones = torch.ones(len(indices), dtype=torch.float32, device=device)
        voxel_flat.scatter_add_(0, indices, ones)

        # 拷回CPU并累加
        voxel_out[:] = voxel_flat.view(2 * num_bins, height, width).cpu().numpy()

        # 立即释放
        del batch_txy, batch_bins, voxel_flat, ones
        if device == 'cuda':
            torch.cuda.empty_cache()


def voxel_to_image(voxel, num_bins):
    """CPU可视化（足够快）"""
    height, width = voxel.shape[1], voxel.shape[2]

    # 时间加权
    weights = np.linspace(0.5, 1.0, num_bins).reshape(-1, 1, 1)
    pos = (voxel[:num_bins] * weights).sum(axis=0)
    neg = (voxel[num_bins:] * weights).sum(axis=0)

    # 对数压缩
    pos_log = np.log1p(pos)
    neg_log = np.log1p(neg)

    max_val = max(pos_log.max(), neg_log.max(), 0.001)

    vis = np.zeros((height, width, 3), dtype=np.uint8)
    vis[:, :, 2] = np.clip(pos_log / max_val * 255, 0, 255).astype(np.uint8)
    vis[:, :, 0] = np.clip(neg_log / max_val * 255, 0, 255).astype(np.uint8)
    vis[:, :, 1] = np.clip(neg_log / max_val * 127, 0, 255).astype(np.uint8)

    # CLAHE
    lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    vis = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return vis


def blend_with_frame(event_img, original_frame):
    """融合原帧"""
    if original_frame.shape[:2] != event_img.shape[:2]:
        original_frame = cv2.resize(original_frame, (event_img.shape[1], event_img.shape[0]))

    gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.4

    return cv2.addWeighted(event_img, 0.9, gray_bgr.astype(np.uint8), 0.3, 0)