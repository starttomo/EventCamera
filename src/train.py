from ultralytics import YOLO

def train():
    model = YOLO('yolov26s.pt')   # 2026最新轻量版，速度+精度最佳
    model.train(
        data='../configs/data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,                  # 你的3060可承受
        device=0,
        name='event_yolov26s',
        pretrained=True,
        optimizer='AdamW',
        augment=True,
        amp=True                   # 混合精度加速
    )
    print("🎉 训练完成！最佳权重在 runs/detect/event_yolov26s/weights/best.pt")