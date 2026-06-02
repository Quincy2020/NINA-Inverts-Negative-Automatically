from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from qnegative.core.models import InvertMode
from qnegative.ui.main_window import MainWindow


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

    window = MainWindow(default_invert_mode=args.invert_mode)
    window.resize(1320, 860)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
