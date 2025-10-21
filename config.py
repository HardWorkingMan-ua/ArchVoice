"""
Модуль для роботи з налаштуваннями
"""

from PyQt6.QtCore import QSettings

class AppConfig:
    """Клас для роботи з налаштуваннями додатку"""

    def __init__(self):
        self.settings = QSettings()

    def save_parameter(self, key, value):
        """Зберегти параметр"""
        self.settings.setValue(key, value)

    def load_parameter(self, key, default=None, type=str):
        """Завантажити параметр"""
        return self.settings.value(key, default, type)

    def save_audio_settings(self, pitch, volume, distortion, echo_delay, echo_feedback, reverb, chorus, noise_gate):
        """Зберегти аудіо налаштування"""
        self.save_parameter("pitch", pitch)
        self.save_parameter("volume", volume)
        self.save_parameter("distortion", distortion)
        self.save_parameter("echo_delay", echo_delay)
        self.save_parameter("echo_feedback", echo_feedback)
        self.save_parameter("reverb", reverb)
        self.save_parameter("chorus", chorus)
        self.save_parameter("noise_gate", noise_gate)

    def load_audio_settings(self):
        """Завантажити аудіо налаштування"""
        return {
            "pitch": self.load_parameter("pitch", 100, int),
            "volume": self.load_parameter("volume", 100, int),
            "distortion": self.load_parameter("distortion", 0, int),
            "echo_delay": self.load_parameter("echo_delay", 0, int),
            "echo_feedback": self.load_parameter("echo_feedback", 0, int),
            "reverb": self.load_parameter("reverb", 0, int),
            "chorus": self.load_parameter("chorus", False, bool),
            "noise_gate": self.load_parameter("noise_gate", 1, int)
        }
