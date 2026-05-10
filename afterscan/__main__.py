import faulthandler
import signal
import sys

from PySide6.QtWidgets import QApplication

from afterscan.app import MainWindow
from afterscan.ui.theme import apply_theme


def main() -> int:
    # `kill -USR1 <pid>` dumps every thread's Python stack — invaluable
    # for diagnosing UI freezes. Cheap to leave on.
    faulthandler.enable()
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    app = QApplication(sys.argv)
    app.setApplicationName("AfterScan")
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
