#!/usr/bin/env python3
"""
Головний файл ArchVoice - Voice Changer для Arch Linux з KDE Plasma
"""

import sys
import subprocess
from PyQt6.QtWidgets import QApplication, QMessageBox

from ui import VoiceChangerMainWindow

def main():
    """Головна функція додатку"""
    app = QApplication(sys.argv)
    app.setApplicationName("ArchVoice")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("ArchVoice")

    # Перевірка залежностей
    missing_deps = []
    try:
        import sounddevice
    except ImportError:
        missing_deps.append("sounddevice (pip install sounddevice)")
    try:
        import scipy
    except ImportError:
        missing_deps.append("scipy (sudo pacman -S python-scipy)")
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy (sudo pacman -S python-numpy)")
    try:
        result = subprocess.run(["pactl", "--version"],
                              capture_output=True, text=True)
        if result.returncode != 0:
            missing_deps.append("PulseAudio (sudo pacman -S pulseaudio)")
    except FileNotFoundError:
        missing_deps.append("PulseAudio (sudo pacman -S pulseaudio)")

    if missing_deps:
        error_msg = "Не знайдено необхідні залежності:\n\n" + "\n".join(missing_deps)
        error_msg += "\n\nВстановіть їх перед запуском додатку."
        QMessageBox.critical(None, "Помилка залежностей", error_msg)
        sys.exit(1)

    window = VoiceChangerMainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
