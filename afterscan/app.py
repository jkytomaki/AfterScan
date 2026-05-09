from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from afterscan import __version__


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"AfterScan — {__version__}")
        self.resize(1440, 900)

        root = QWidget()
        root.setObjectName("stage")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        placeholder = QLabel("AfterScan modern UI — phase 0 shell")
        placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(placeholder, 1)
