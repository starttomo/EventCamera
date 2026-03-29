以下是优化后的 Markdown 格式，结构更清晰，便于阅读和使用：

```markdown
# EventCamera 项目结构

```
EventCamera/
├── data/
│   ├── raw/                      # UA-DETRAC 视频 + XML 标注
│   └── processed/
│       ├── events/               # .csv 或 .npz 格式的事件数据
│       ├── event_frames/         # 优化后的事件图像/张量
│       └── labels/               # YOLO 格式标注文件 (.txt)
│
├── src/
│   ├── event_simulator.py        # 升级版模拟器（GPU 加速 + 更好表征）
│   ├── preprocess.py             # 事件 → 帧 + 标注转换
│   ├── train.py                  # YOLOv8 训练脚本
│   ├── detect.py                 # 推理脚本
│   └── compare_visual.py         # GPU 加速同屏对比可视化
│
├── configs/
│   └── data.yaml                 # 数据集配置文件
│
├── runs/                         # 训练权重、日志、结果
├── models/                       # 导出的模型文件
├── main.py                       # 一键运行流程
└── event_camera_gui.py           # GUI 图形界面
```

## 功能说明

| 模块 | 说明 |
|------|------|
| `event_simulator.py` | 事件相机模拟器，支持 GPU 加速与多种事件表征 |
| `preprocess.py` | 将事件数据转换为帧图像，并同步转换标注格式 |
| `train.py` | 使用 YOLOv8 进行模型训练 |
| `detect.py` | 加载模型进行推理检测 |
| `compare_visual.py` | 同屏对比可视化，支持 GPU 加速渲染 |
| `main.py` | 串联完整流程（模拟 → 预处理 → 训练 → 推理） |
| `event_camera_gui.py` | 图形界面，便于交互操作 |
```