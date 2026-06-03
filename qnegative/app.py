from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QProgressBar, QSplashScreen

from qnegative.core.models import InvertMode


def _create_splash() -> tuple[QSplashScreen, QProgressBar]:
    pixmap = QPixmap(440, 180)
    pixmap.fill(QColor("#15181d"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("#f2f4f7"))
    painter.drawText(28, 54, "QNegativeLab")
    painter.setPen(QColor("#9aa4b2"))
    painter.drawText(28, 82, "Loading film workspace...")
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    bar = QProgressBar(splash)
    bar.setGeometry(28, 124, 384, 18)
    bar.setRange(0, 100)
    bar.setValue(8)
    bar.setTextVisible(False)
    bar.setStyleSheet(
        """
        QProgressBar {
            background: #20242b;
            border: 1px solid #343c47;
            border-radius: 5px;
        }
        QProgressBar::chunk {
            background: #4aa3ff;
            border-radius: 4px;
        }
        """
    )
    return splash, bar


def main() -> int:
    parser = argparse.ArgumentParser(description="QNegativeLab")
    parser.add_argument(
        "--invert-mode",
        choices=[
            InvertMode.DENSITY.value,
            InvertMode.LAB_PRINT.value,
            InvertMode.LOG_BOUNDS.value,
            InvertMode.SIMPLE.value,
        ],
        default=InvertMode.LAB_PRINT.value,
        help="Default inversion model for new images.",
    )
    args, qt_args = parser.parse_known_args(sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("QNegativeLab")
    app.setOrganizationName("QNegativeLab")

    splash, splash_bar = _create_splash()
    splash.show()
    app.processEvents()

    splash_bar.setValue(35)
    splash.showMessage("Loading UI modules...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#cfd6df"))
    app.processEvents()
    from qnegative.ui.main_window import MainWindow

    splash_bar.setValue(70)
    splash.showMessage("Building main window...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#cfd6df"))
    app.processEvents()
    window = MainWindow(default_invert_mode=args.invert_mode)
    window.resize(1320, 860)
    splash_bar.setValue(95)
    splash.showMessage("Ready", Qt.AlignBottom | Qt.AlignHCenter, QColor("#cfd6df"))
    app.processEvents()
    window.show()
    splash_bar.setValue(100)
    splash.finish(window)

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
