from __future__ import annotations

import argparse
import sys

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication, QProgressBar, QSplashScreen

from qnegative.core.diagnostics import enable_crash_logging, install_qt_message_logger, log_event
from qnegative.resources import resource_path


def _create_splash() -> tuple[QSplashScreen, QProgressBar]:
    pixmap = QPixmap(600, 240)
    pixmap.fill(QColor("#1A1A1A"))
    title_path = resource_path("logo/NINA_TITLE.svg")
    painter = QPainter(pixmap)
    if title_path.exists():
        renderer = QSvgRenderer(str(title_path))
        renderer.render(painter, QRectF(0, 0, 600, 200))
    else:
        painter.setPen(QColor("#FFB000"))
        painter.drawText(75, 120, "NINA")
    painter.end()

    splash = QSplashScreen(pixmap)
    splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    bar = QProgressBar(splash)
    bar.setGeometry(75, 214, 450, 12)
    bar.setRange(0, 100)
    bar.setValue(8)
    bar.setTextVisible(False)
    bar.setStyleSheet(
        """
        QProgressBar {
            background: #121212;
            border: 1px solid #444444;
            border-radius: 5px;
        }
        QProgressBar::chunk {
            background: #FFB000;
            border-radius: 4px;
        }
        """
    )
    return splash, bar


def main() -> int:
    log_path = enable_crash_logging()
    print(f"NINA crash log: {log_path}", flush=True)
    parser = argparse.ArgumentParser(description="NINA - NINA Inverts Negative Automatically")
    _, qt_args = parser.parse_known_args(sys.argv[1:])

    app = QApplication([sys.argv[0], *qt_args])
    install_qt_message_logger()
    app.setApplicationName("NINA")
    app.setOrganizationName("NINA")
    logo_path = resource_path("logo/NINA_LOGO.svg")
    if logo_path.exists():
        app.setWindowIcon(QIcon(str(logo_path)))

    splash, splash_bar = _create_splash()
    splash.show()
    log_event("app", "Splash shown")
    app.processEvents()

    splash_bar.setValue(35)
    splash.showMessage("Loading UI modules...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#D8D0C2"))
    app.processEvents()
    from qnegative.ui.main_window import MainWindow

    splash_bar.setValue(70)
    splash.showMessage("Building main window...", Qt.AlignBottom | Qt.AlignHCenter, QColor("#D8D0C2"))
    app.processEvents()
    window = MainWindow()
    log_event("app", "MainWindow constructed")
    window.resize(1320, 860)
    splash_bar.setValue(95)
    splash.showMessage("Ready", Qt.AlignBottom | Qt.AlignHCenter, QColor("#D8D0C2"))
    app.processEvents()
    window.show()
    splash_bar.setValue(100)
    splash.finish(window)
    log_event("app", "Entering Qt event loop")

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
