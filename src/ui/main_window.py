import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QKeyEvent
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QHBoxLayout, QVBoxLayout,
    QDoubleSpinBox, QFormLayout, QGroupBox, QCheckBox, QSpinBox
)
from qasync import asyncSlot

from capture.ir_capture import IRCapture
from detect.drowsy import IRDrowsyDetector
from utils.image import numpy_to_qimage

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IR MVP (OOP): Torch + Drowsy ON/OFF + Press C to save PNG")
        self.resize(1000, 660)

        # Core components
        self.cap = IRCapture()
        self.det = IRDrowsyDetector()

        # Preview
        self.label = QLabel("IR Preview")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumSize(720, 540)

        # Status
        self.lbl_status = QLabel("상태: -")
        self._last_vis: Optional = None

        # Drowsy
        self.chk_drowsy = QCheckBox("Enable IR Drowsy Detection")
        self.chk_drowsy.setChecked(True)
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.10, 0.40); self.spin_thresh.setSingleStep(0.01); self.spin_thresh.setValue(0.22)

        # Torch
        self.chk_torch = QCheckBox("Enable IR Torch")
        self.spin_power = QSpinBox(); self.spin_power.setRange(0, 100); self.spin_power.setValue(50)
        self.btn_apply_torch = QPushButton("Apply Torch")

        # Controls
        self.btn_start = QPushButton("Start")
        self.btn_stop  = QPushButton("Stop")

        # Layout
        top = QHBoxLayout()
        top.addWidget(self.btn_start)
        top.addWidget(self.btn_stop)
        top.addStretch()
        top.addWidget(self.lbl_status)

        cfg_d = QGroupBox("Drowsy")
        f = QFormLayout()
        f.addRow(self.chk_drowsy)
        f.addRow("EAR Threshold", self.spin_thresh)
        cfg_d.setLayout(f)

        cfg_t = QGroupBox("IR Torch")
        ft = QFormLayout()
        ft.addRow(self.chk_torch)
        ft.addRow("Power (0~100)", self.spin_power)
        ft.addRow(self.btn_apply_torch)
        cfg_t.setLayout(ft)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self.label, 1)
        bottom = QHBoxLayout()
        bottom.addWidget(cfg_d)
        bottom.addWidget(cfg_t)
        bottom.addStretch()
        root.addLayout(bottom)

        # Signals
        self.btn_start.clicked.connect(self._start_clicked)
        self.btn_stop.clicked.connect(self._stop_clicked)
        self.btn_apply_torch.clicked.connect(self._apply_torch_clicked)
        self.chk_torch.toggled.connect(self._apply_torch_clicked)
        self.spin_power.valueChanged.connect(self._apply_torch_clicked)

        # Timer
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        # Save folder
        self._save_dir = Path("result"); self._save_dir.mkdir(exist_ok=True)

    # --- Keybind: Space to save ---
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_C:
            self._save_current_frame()
        else:
            super().keyPressEvent(event)

    def _save_current_frame(self):
        if self._last_vis is None:
            self.lbl_status.setText("상태: 저장할 이미지가 없습니다.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self._save_dir / f"ir_{ts}.png"
        ok = cv2.imwrite(str(path), self._last_vis)
        self.lbl_status.setText(f"상태: Saved {path.name}" if ok else "상태: 저장 실패")

    # --- Buttons ---
    @asyncSlot()
    async def _start_clicked(self):
        ok = await self.cap.start()
        if not ok:
            self.label.setText("IR 소스를 찾을 수 없습니다.")
            self.lbl_status.setText("상태: Start 실패")
        else:
            self.label.setText("IR 시작됨 (Space=저장)")
            self.lbl_status.setText("상태: Start 성공")
            await self._apply_torch_internal()  # apply current torch state

    @asyncSlot()
    async def _stop_clicked(self):
        await self.cap.stop()
        self.label.setText("정지됨")
        self.lbl_status.setText("상태: Stop")
        self._last_vis = None

    @asyncSlot()
    async def _apply_torch_clicked(self):
        await self._apply_torch_internal()

    async def _apply_torch_internal(self):
        enable = self.chk_torch.isChecked()
        power = int(self.spin_power.value())
        ok, msg = await self.cap.set_torch(enable, power)
        self.lbl_status.setText(f"상태: {msg}")

    # --- UI Loop ---
    def _tick(self):
        gray = self.cap.last_gray
        if gray is None:
            return

        if self.chk_drowsy.isChecked():
            ear, vis = self.det.process(gray, float(self.spin_thresh.value()))
            if vis is None:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        self._last_vis = vis.copy()
        qimg = numpy_to_qimage(vis)
        self.label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
