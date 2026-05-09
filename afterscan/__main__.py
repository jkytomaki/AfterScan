import sys

from PySide6.QtWidgets import QApplication

from afterscan.app import MainWindow
from afterscan.ui.theme import apply_theme


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("AfterScan")
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
