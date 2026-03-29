import sys
import cv2
import os
from src.event_simulator import EventSimulator
from src.preprocess import events_to_voxel_grid
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QGroupBox,
    QSlider, QStyle, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt, QUrl
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO
from src.compare_visual import compare  # 如果你想保留原对比也可，但这里我们用新GUI
from src.event_simulator import EventSimulator

# ====================== 导入你的预处理函数 ======================
from src.preprocess import events_to_voxel_grid   # ← 确保你的 preprocess.py 有这个函数

class EventCameraGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("事件相机 vs 传统帧相机 实时对比系统")
        self.resize(1600, 900)

        self.rgb_model = YOLO('yolo26s.pt')      # RGB 检测模型
        # self.event_model = YOLO('your_event_model.pt')  # 以后可以加上事件检测模型

        self.cap = None
        self.voxel_files = []
        self.frame_idx = 0
        self.is_playing = False
        self.video_path = None
        self.voxel_dir = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)

        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ==================== 左侧：两个视频播放区 ====================
        video_layout = QVBoxLayout()

        # RGB 视频
        rgb_group = QGroupBox("传统帧相机 (RGB + YOLO26s 检测)")
        rgb_layout = QVBoxLayout()
        self.rgb_label = QLabel()
        self.rgb_label.setAlignment(Qt.AlignCenter)
        self.rgb_label.setMinimumSize(800, 450)
        self.rgb_label.setStyleSheet("background-color: black;")
        rgb_layout.addWidget(self.rgb_label)
        rgb_group.setLayout(rgb_layout)

        # Event 视频
        event_group = QGroupBox("模拟事件相机 (Voxel Grid)")
        event_layout = QVBoxLayout()
        self.event_label = QLabel()
        self.event_label.setAlignment(Qt.AlignCenter)
        self.event_label.setMinimumSize(800, 450)
        self.event_label.setStyleSheet("background-color: black;")
        event_layout.addWidget(self.event_label)
        event_group.setLayout(event_layout)

        video_layout.addWidget(rgb_group)
        video_layout.addWidget(event_group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.btn_select = QPushButton("📂 选择视频")
        self.btn_select.clicked.connect(self.select_video)
        self.btn_play = QPushButton("▶ 播放 / 暂停")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_reset = QPushButton("🔄 重置")
        self.btn_reset.clicked.connect(self.reset_video)

        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_reset)
        video_layout.addLayout(btn_layout)

        main_layout.addLayout(video_layout, stretch=7)

        # ==================== 右侧：信息面板 ====================
        info_layout = QVBoxLayout()
        info_group = QGroupBox("检测信息 & 车辆轨迹趋势")
        info_inner = QVBoxLayout()

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumWidth(380)
        info_inner.addWidget(self.info_text)

        # 车辆数量实时显示
        self.vehicle_count_label = QLabel("当前车辆数: 0")
        self.vehicle_count_label.setStyleSheet("font-size: 18px; color: lime; font-weight: bold;")
        info_inner.addWidget(self.vehicle_count_label)

        # 轨迹趋势（简单文字 + 后期可扩展）
        self.trajectory_label = QLabel("轨迹趋势: 待检测...")
        self.trajectory_label.setStyleSheet("font-size: 16px; color: cyan;")
        info_inner.addWidget(self.trajectory_label)

        info_group.setLayout(info_inner)
        info_layout.addWidget(info_group)

        # 日志输出
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        info_layout.addWidget(log_group)

        main_layout.addLayout(info_layout, stretch=3)

        self.log("✅ PyQt5 前端初始化完成，点击「选择视频」开始")

    def log(self, message):
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def select_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "data/raw", "Video Files (*.mp4 *.avi *.mov)")
        if not file:
            return

        self.video_path = Path(file)
        video_stem = self.video_path.stem
        self.voxel_dir = Path("data/processed/event_frames") / video_stem
        self.events_npz = Path("data/processed/events") / f"{video_stem}.npz"

        self.log(f"📂 已选择视频: {self.video_path.name}")

        # 检查缓存
        if self.voxel_dir.exists() and len(list(self.voxel_dir.glob("vis_*.jpg"))) > 5:
            self.log("✅ 检测到已处理的 Voxel Grid，直接加载缓存")
        else:
            self.log("⚙️ 开始处理视频（事件模拟 + Voxel Grid 生成）...")
            try:
                os.makedirs(self.voxel_dir.parent, exist_ok=True)
                os.makedirs(self.events_npz.parent, exist_ok=True)

                # 第一步：生成事件 .npz
                if not self.events_npz.exists():
                    self.log("   → 正在模拟事件数据...")
                    # ==================== 关键修改：降低阈值，启用帧插值 ====================
                    simulator = EventSimulator(
                        threshold=0.035,  # 从0.05降到0.02
                        noise_threshold=0.015  # 添加噪声过滤
                    )
                    simulator.simulate(
                        str(self.video_path),
                        str(self.events_npz),
                        fps_boost=4  # 4倍帧插值
                    )
                    self.log("   ✅ 事件 .npz 生成完成")
                else:
                    self.log("   ✅ 已存在事件 .npz，跳过模拟")

                # 第二步：生成 Voxel Grid
                self.log("   → 正在生成 Voxel Grid...")
                # ==================== 关键修改：优化可视化参数 ====================
                events_to_voxel_grid(
                    str(self.events_npz),
                    str(self.voxel_dir),
                    num_bins=10,  # 从6增加到10
                    accumulation_time=0.03  # 从0.05减少到0.03
                )
                self.log("✅ Voxel Grid 生成完成！")

            except Exception as e:
                self.log(f"❌ 处理失败: {e}")
                import traceback
                self.log(traceback.format_exc())
                return

        # 加载视频和事件可视化帧
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.voxel_files = sorted(self.voxel_dir.glob("vis_*.jpg"))

        if not self.voxel_files:
            self.log("❌ 未找到 vis_*.jpg 文件，请检查 preprocess.py")
            return

        self.frame_idx = 0
        self.log(f"🎥 准备播放，共 {len(self.voxel_files)} 帧")
        self.toggle_play()

    def toggle_play(self):
        if self.cap is None:
            self.log("⚠️ 请先选择视频")
            return

        self.is_playing = not self.is_playing
        if self.is_playing:
            self.btn_play.setText("⏸ 暂停")
            self.timer.start(33)   # ≈30 FPS
        else:
            self.btn_play.setText("▶ 播放")
            self.timer.stop()

    def reset_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.frame_idx = 0
        self.is_playing = False
        self.timer.stop()
        self.btn_play.setText("▶ 播放")
        self.log("🔄 已重置到视频开头")

    def update_frames(self):
        if self.cap is None or not self.is_playing:
            return

        ret, rgb_frame = self.cap.read()
        if not ret or self.frame_idx >= len(self.voxel_files):
            self.log("✅ 视频播放结束")
            self.toggle_play()
            return

        # 读取事件帧
        event_path = self.voxel_files[self.frame_idx]
        event_img = cv2.imread(str(event_path))

        # ====================== YOLO 检测（RGB 侧） ======================
        results = self.rgb_model(rgb_frame, verbose=False)
        rgb_annotated = results[0].plot() if results else rgb_frame.copy()

        # ====================== 信息统计 ======================
        vehicle_count = len(results[0].boxes) if results and results[0].boxes else 0
        self.vehicle_count_label.setText(f"当前车辆数: {vehicle_count}")

        # 简单轨迹趋势（后续可扩展为真实轨迹预测）
        trend = "→ 向右平稳移动" if vehicle_count > 0 else "无车辆"
        self.trajectory_label.setText(f"轨迹趋势: {trend}")

        # ====================== 显示到 QLabel ======================
        def cv2_to_pixmap(img):
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            return QPixmap.fromImage(qimg)

        self.rgb_label.setPixmap(cv2_to_pixmap(rgb_annotated).scaled(800, 450, Qt.KeepAspectRatio))
        self.event_label.setPixmap(cv2_to_pixmap(event_img).scaled(800, 450, Qt.KeepAspectRatio))

        self.frame_idx += 1

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EventCameraGUI()
    window.show()
    sys.exit(app.exec_())