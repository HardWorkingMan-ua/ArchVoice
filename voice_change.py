#!/usr/bin/env python3
"""
Voice Changer для Arch Linux з KDE Plasma
Створює віртуальний мікрофон з обробкою голосу
"""

import sys
import os
import subprocess
import json
import numpy as np
import sounddevice as sd
import threading
import time
from scipy import signal
from scipy.io import wavfile
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QSlider, QLabel, QComboBox,
                             QGroupBox, QGridLayout, QSpinBox, QCheckBox, QFrame,
                             QMessageBox, QStatusBar, QProgressBar, QListWidget,
                             QListWidgetItem, QSplitter, QTextEdit, QTabWidget)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QSettings
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor, QLinearGradient, QGradient

def reload_audio_devices():
    """Перезавантаження звукової системи для оновлення списку пристроїв"""
    try:
        sd._terminate()
        sd._initialize()
        return sd.query_devices()
    except Exception as e:
        print(f"Помилка перезавантаження звукової системи: {e}")
        return sd.query_devices()

class VirtualMicrophoneManager:
    """Менеджер віртуального мікрофону через PulseAudio"""

    def __init__(self):
        self.virtual_sink_name = "voice_changer_sink"
        self.virtual_source_name = "voice_changer_source"
        self.loopback_module_id = None

    def create_virtual_devices(self):
        """Створення віртуальних аудіо пристроїв"""
        try:
            cmd = [
                "pactl", "load-module", "module-null-sink",
                f"sink_name={self.virtual_sink_name}",
                f"sink_properties=device.description='Voice_Changer_Output'"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, f"Помилка створення sink: {result.stderr}"

            cmd = [
                "pactl", "load-module", "module-remap-source",
                f"source_name={self.virtual_source_name}",
                f"master={self.virtual_sink_name}.monitor",
                f"source_properties=device.description='Voice_Changer_Microphone'"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, f"Помилка створення source: {result.stderr}"
            return True, "Віртуальні пристрої створено успішно"
        except Exception as e:
            return False, f"Помилка: {e}"

    def remove_virtual_devices(self):
        """Видалення віртуальних аудіо пристроїв"""
        try:
            result = subprocess.run(["pactl", "list", "modules", "short"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if self.virtual_sink_name in line or self.virtual_source_name in line:
                        module_id = line.split('\t')[0]
                        subprocess.run(["pactl", "unload-module", module_id])
        except Exception as e:
            print(f"Помилка видалення віртуальних пристроїв: {e}")

class AudioProcessor(QObject):
    """Клас для обробки аудіо в реальному часі"""

    level_updated = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.is_processing = False
        self.sample_rate = None
        self.block_size = 512
        self.pitch_shift = 1.0
        self.formant_shift = 1.0
        self.reverb_amount = 0.0
        self.echo_delay = 0.0
        self.echo_feedback = 0.0
        self.volume = 1.0
        self.distortion = 0.0
        self.chorus_enabled = False
        self.noise_gate_threshold = 0.01
        self.low_pass_freq = 8000
        self.high_pass_freq = 80
        self.delay_buffer = None
        self.reverb_buffer = None
        self.output_buffer = None
        self.buffer_write_index = 0
        self.buffer_read_index = 0
        self.buffer_lock = threading.Lock()

    def setup_filters(self):
        """Налаштування аудіо фільтрів"""
        try:
            nyquist = self.sample_rate / 2
            self.lp_b, self.lp_a = signal.butter(4, self.low_pass_freq / nyquist, 'low')
            self.lp_zi = signal.lfilter_zi(self.lp_b, self.lp_a)
            self.hp_b, self.hp_a = signal.butter(4, self.high_pass_freq / nyquist, 'high')
            self.hp_zi = signal.lfilter_zi(self.hp_b, self.hp_a)
            print("Filters initialized successfully")
        except Exception as e:
            print(f"Error setting up filters: {e}")

    def start_processing(self, input_device_id, virtual_sink_name):
        """Запуск обробки аудіо з виводом у віртуальний sink"""
        try:
            self.is_processing = True
            input_info = sd.query_devices(input_device_id)

            # Виводимо всі пристрої для дебагу
            print("\nДоступні пристрої звуку:")
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                input_ch = device['max_input_channels']
                output_ch = device['max_output_channels']
                flags = []
                if input_ch > 0: flags.append("Мікрофон")
                if output_ch > 0: flags.append("Вихід")
                flags_str = " | ".join(flags)
                print(f"{i}: {device['name']} [{flags_str}]")

            # Шукаємо віртуальний пристрій
            virtual_sink_id = None
            for i, device in enumerate(devices):
                if (device['max_output_channels'] > 0 and
                    any(keyword.lower() in device['name'].lower()
                        for keyword in [virtual_sink_name, "Voice_Changer_Output", "voice_changer_sink"])):
                    virtual_sink_id = i
                    print(f"Знайдено віртуальний пристрій: {device['name']} (ID: {i})")
                    break

            if virtual_sink_id is None:
                return False, "Віртуальний sink не знайдено. Переконайтесь, що він створений."

            print(f"Вхід: {input_info['name']}")
            print(f"Вихід: {sd.query_devices(virtual_sink_id)['name']}")

            # Виправлений виклик методу
            return self.start_pipewire_processing(input_device_id, virtual_sink_id)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Помилка запуску: {e}"

    def find_virtual_sink_with_pactl(self):
        """Знаходить ID вихідного пристрою за допомогою pactl"""
        try:
            result = subprocess.run(["pactl", "list", "short", "sinks"],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return None
            for line in result.stdout.split('\n'):
                if "voice_changer_sink" in line or "Voice_Changer_Output" in line:
                    parts = line.split('\t')
                    if len(parts) > 1:
                        sink_name = parts[1]
                        devices = sd.query_devices()
                        for i, device in enumerate(devices):
                            if sink_name in device['name'] and device['max_output_channels'] > 0:
                                print(f"Знайдено віртуальний пристрій через pactl: {device['name']} (ID: {i})")
                                return i
            return None
        except Exception:
            return None

    def start_pipewire_processing(self, input_device_id, virtual_sink_id):
        """Спеціальний спосіб запуску для PipeWire"""
        try:
            input_info = sd.query_devices(input_device_id)
            output_info = sd.query_devices(virtual_sink_id)
            self.sample_rate = int(input_info['default_samplerate'])
            input_channels = min(input_info['max_input_channels'], 2)
            output_channels = min(output_info['max_output_channels'], 2)
            print(f"Input sample rate: {input_info['default_samplerate']}, Output sample rate: {output_info['default_samplerate']}")
            print(f"Using sample rate: {self.sample_rate} Hz")
            print(f"Input channels: {input_channels}, Output channels: {output_channels}")

            self.setup_filters()
            self.delay_buffer = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            self.reverb_buffer = np.zeros(int(self.sample_rate * 0.3), dtype=np.float32)
            self.output_buffer = np.zeros((self.block_size * 100, 2), dtype=np.float32)
            self.buffer_write_index = 0
            self.buffer_read_index = 0
            print(f"Initialized output_buffer with shape: {self.output_buffer.shape}")

            print("Creating input stream...")
            self.input_stream = sd.InputStream(
                device=input_device_id,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=input_channels,
                dtype=np.float32,
                callback=self.input_callback
            )

            print("Creating output stream...")
            self.output_stream = sd.OutputStream(
                device=virtual_sink_id,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                channels=output_channels,
                dtype=np.float32,
                callback=self.output_callback
            )

            print("Starting input stream...")
            self.input_stream.start()
            time.sleep(0.5)
            print("Starting output stream...")
            self.output_stream.start()
            print(f"Output stream active: {self.output_stream.active}")
            print("Audio streams started")
            return True, "Обробка запущена успішно"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Помилка запуску: {e}"

    def input_callback(self, indata, frames, time, status):
        """Callback для вхідного потоку"""
        if not self.is_processing:
            return  # Не обробляємо, якщо обробка зупинена

        if status:
            print(f"Input status: {status}")
        if len(indata) == 0:
            return

        input_audio = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
        if len(input_audio) != self.block_size:
            input_audio = np.pad(input_audio, (0, self.block_size - len(input_audio)), mode='constant')[:self.block_size]

        level = np.sqrt(np.mean(input_audio**2))
        self.level_updated.emit(level)

        processed_audio = self.process_audio(input_audio)
        processed_stereo = np.column_stack((processed_audio, processed_audio))

        with self.buffer_lock:
            end_idx = self.buffer_write_index + frames
            if end_idx > len(self.output_buffer):
                self.buffer_write_index = 0
                end_idx = frames
            self.output_buffer[self.buffer_write_index:end_idx] = processed_stereo
            self.buffer_write_index = end_idx

    def output_callback(self, outdata, frames, time, status):
        """Callback для вихідного потоку"""
        if not self.is_processing:
            outdata.fill(0)  # Заповнюємо нулями
            return

        if status:
            print(f"Output status: {status}")

        with self.buffer_lock:
            available_frames = self.buffer_write_index - self.buffer_read_index
            if available_frames < frames:
                outdata.fill(0)
                return

            outdata[:] = self.output_buffer[self.buffer_read_index:self.buffer_read_index + frames]
            self.buffer_read_index += frames

            if self.buffer_read_index >= self.buffer_write_index:
                self.buffer_read_index = 0
                self.buffer_write_index = 0

    def apply_pitch_shift(self, audio_data):
        """Застосування зміни висоти тону"""
        if self.pitch_shift == 1.0:
            return audio_data

        n = len(audio_data)
        f = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        shifted_f = np.zeros_like(f)
        new_n = int(n / self.pitch_shift)
        new_freqs = np.linspace(0, max(freqs), len(shifted_f))

        for i, freq in enumerate(new_freqs):
            if freq * self.pitch_shift < max(freqs):
                idx = np.argmin(np.abs(freqs - freq * self.pitch_shift))
                shifted_f[i] = f[idx]

        shifted_audio = np.fft.irfft(shifted_f)
        if len(shifted_audio) < n:
            shifted_audio = np.pad(shifted_audio, (0, n - len(shifted_audio)), 'constant')
        elif len(shifted_audio) > n:
            shifted_audio = shifted_audio[:n]
        return shifted_audio

    def process_audio(self, audio_data):
        """Основна функція обробки аудіо"""
        print(f"Processing with: Pitch={self.pitch_shift}, Distortion={self.distortion}, Echo={self.echo_delay}, Reverb={self.reverb_amount}, Chorus={self.chorus_enabled}")
        if len(audio_data) != self.block_size:
            print(f"Warning: audio_data length {len(audio_data)} does not match block_size {self.block_size}")
            audio_data = np.pad(audio_data, (0, self.block_size - len(audio_data)), mode='constant')[:self.block_size]

        processed = audio_data.copy()
        print(f"Input audio max amplitude: {np.max(np.abs(processed))}")

        # Noise gate
        if np.max(np.abs(processed)) < self.noise_gate_threshold:
            processed = np.zeros_like(processed)
            print("Noise gate applied: signal below threshold")

        # High-pass filter
        processed, self.hp_zi = signal.lfilter(self.hp_b, self.hp_a, processed, zi=self.hp_zi)
        print(f"After high-pass filter: {np.max(np.abs(processed))}")

        # Pitch shift
        processed = self.apply_pitch_shift(processed)
        print(f"After pitch shift: {np.max(np.abs(processed))}")

        # Distortion
        if self.distortion > 0:
            processed = self.apply_distortion(processed)
            print(f"After distortion: {np.max(np.abs(processed))}")

        # Echo
        if self.echo_delay > 0 and self.echo_feedback > 0:
            processed = self.apply_echo(processed)
            print(f"After echo: {np.max(np.abs(processed))}")

        # Reverb
        if self.reverb_amount > 0:
            processed = self.apply_reverb(processed)
            print(f"After reverb: {np.max(np.abs(processed))}")

        # Chorus
        if self.chorus_enabled:
            processed = self.apply_chorus(processed)
            print(f"After chorus: {np.max(np.abs(processed))}")

        # Low-pass filter
        processed, self.lp_zi = signal.lfilter(self.lp_b, self.lp_a, processed, zi=self.lp_zi)
        print(f"After low-pass filter: {np.max(np.abs(processed))}")

        # Volume adjustment
        processed *= self.volume
        print(f"After volume adjustment: {np.max(np.abs(processed))}")

        # Clipping
        processed = np.clip(processed, -0.95, 0.95)
        print(f"Final output max amplitude: {np.max(np.abs(processed))}")

        return processed

    def apply_distortion(self, audio_data):
        """Застосування дисторшну"""
        drive = 1.0 + self.distortion * 5
        distorted = audio_data * drive
        distorted = np.tanh(distorted * 0.7)
        return distorted * (1.0 - self.distortion * 0.3)

    def apply_echo(self, audio_data):
        """Посилене ехо"""
        delay_samples = int(self.echo_delay * self.sample_rate)
        if delay_samples >= len(self.delay_buffer) or delay_samples <= 0:
            print(f"Echo skipped: delay_samples={delay_samples}, buffer_size={len(self.delay_buffer)}")
            return audio_data
        output = audio_data.copy()
        self.delay_buffer = np.roll(self.delay_buffer, len(audio_data))
        self.delay_buffer[:len(audio_data)] = output + self.delay_buffer[:len(audio_data)] * self.echo_feedback * 0.8
        delayed = self.delay_buffer[delay_samples:delay_samples + len(audio_data)]
        output += delayed * self.echo_feedback * 0.8
        return output

    def apply_reverb(self, audio_data):
        """Посилена реверберація"""
        delays = [0.03, 0.032, 0.035, 0.038]
        reverb_output = audio_data.copy()
        for delay in delays:
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(self.reverb_buffer):
                delayed = np.roll(self.reverb_buffer, delay_samples)[:len(audio_data)]
                reverb_output += delayed * self.reverb_amount * 0.5
        self.reverb_buffer = np.roll(self.reverb_buffer, len(audio_data))
        self.reverb_buffer[:len(audio_data)] = reverb_output
        return reverb_output * (1.0 + self.reverb_amount * 0.5)

    def apply_chorus(self, audio_data):
        """Посилений хорус"""
        t = np.arange(len(audio_data)) / self.sample_rate
        lfo = np.sin(2 * np.pi * 0.5 * t) * 0.01
        chorus_output = audio_data.copy()
        base_delay = int(0.02 * self.sample_rate)
        for i in range(len(audio_data)):
            delay_mod = int(base_delay + lfo[i] * self.sample_rate)
            if i >= delay_mod:
                chorus_output[i] += audio_data[i - delay_mod] * 0.7
        return chorus_output * 0.8

    def stop_processing(self):
        """Зупинка обробки аудіо"""
        self.is_processing = False
        print("Запит на зупинку обробки...")

        # Даємо трохи часу для завершення колбеків
        time.sleep(0.1)

        # Зупиняємо потоки
        if hasattr(self, 'input_stream'):
            try:
                if self.input_stream.active:
                    print("Зупиняємо input stream...")
                    self.input_stream.stop()
                if self.input_stream:
                    self.input_stream.close()
                print("Input stream зупинено та закрито")
            except Exception as e:
                print(f"Помилка при зупинці input stream: {e}")

        if hasattr(self, 'output_stream'):
            try:
                if self.output_stream.active:
                    print("Зупиняємо output stream...")
                    self.output_stream.stop()
                if self.output_stream:
                    self.output_stream.close()
                print("Output stream зупинено та закрито")
            except Exception as e:
                print(f"Помилка при зупинці output stream: {e}")

        # Очищаємо буфери
        with self.buffer_lock:
            if self.output_buffer is not None:
                self.output_buffer.fill(0)
            self.buffer_write_index = 0
            self.buffer_read_index = 0
            print("Буфери очищено")

        print("Обробка повністю зупинена")

class VoicePresets:
    """Менеджер пресетів голосу"""

    @staticmethod
    def get_presets():
        return {
            "Оригінал": {
                "pitch_shift": 1.0,
                "volume": 1.0,
                "distortion": 0.0,
                "echo_delay": 0.0,
                "echo_feedback": 0.0,
                "reverb_amount": 0.0,
                "chorus_enabled": False,
                "description": "Без обробки"
            },
            "Чоловічий голос": {
                "pitch_shift": 0.85,
                "volume": 1.1,
                "distortion": 0.05,
                "echo_delay": 0.0,
                "echo_feedback": 0.0,
                "reverb_amount": 0.1,
                "chorus_enabled": False,
                "description": "Низький чоловічий голос"
            },
            "Жіночий голос": {
                "pitch_shift": 1.3,
                "volume": 0.9,
                "distortion": 0.0,
                "echo_delay": 0.0,
                "echo_feedback": 0.0,
                "reverb_amount": 0.05,
                "chorus_enabled": False,
                "description": "Високий жіночий голос"
            },
            "Дитячий голос": {
                "pitch_shift": 1.6,
                "volume": 0.8,
                "distortion": 0.0,
                "echo_delay": 0.0,
                "echo_feedback": 0.0,
                "reverb_amount": 0.0,
                "chorus_enabled": True,
                "description": "Дитячий голос"
            },
            "Робот": {
                "pitch_shift": 0.9,
                "volume": 1.0,
                "distortion": 0.3,
                "echo_delay": 0.05,
                "echo_feedback": 0.2,
                "reverb_amount": 0.0,
                "chorus_enabled": False,
                "description": "Механічний робот"
            },
            "Демон": {
                "pitch_shift": 0.7,
                "volume": 1.2,
                "distortion": 0.4,
                "echo_delay": 0.1,
                "echo_feedback": 0.3,
                "reverb_amount": 0.4,
                "chorus_enabled": False,
                "description": "Демонічний голос"
            },
            "Радіо диктор": {
                "pitch_shift": 0.95,
                "volume": 1.0,
                "distortion": 0.1,
                "echo_delay": 0.0,
                "echo_feedback": 0.0,
                "reverb_amount": 0.15,
                "chorus_enabled": False,
                "description": "Голос радіо диктора"
            },
            "Космічний ефект": {
                "pitch_shift": 1.1,
                "volume": 0.9,
                "distortion": 0.2,
                "echo_delay": 0.15,
                "echo_feedback": 0.4,
                "reverb_amount": 0.6,
                "chorus_enabled": True,
                "description": "Космічний ефект"
            },
            "Телефон": {
                "pitch_shift": 1.0,
                "volume": 0.8,
                "distortion": 0.15,
                "echo_delay": 0.0,
                "echo_feedback": 0.0,
                "reverb_amount": 0.0,
                "chorus_enabled": False,
                "description": "Звук через телефон"
            },
            "Печера": {
                "pitch_shift": 0.9,
                "volume": 1.0,
                "distortion": 0.0,
                "echo_delay": 0.2,
                "echo_feedback": 0.5,
                "reverb_amount": 0.8,
                "chorus_enabled": False,
                "description": "Відлуння в печері"
            }
        }

class VoiceChangerMainWindow(QMainWindow):
    """Головне вікно додатку"""
    processing_stopped = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.audio_processor = AudioProcessor()
        self.virtual_mic = VirtualMicrophoneManager()
        self.is_running = False
        self.setWindowIcon(QIcon("/usr/share/icons/hicolor/256x256/apps/archvoice.png"))
        self.settings = QSettings()

        self.init_ui()
        self.setup_audio_devices()
        self.setup_virtual_devices()

        self.audio_processor.level_updated.connect(self.update_level_indicator)
        self.load_settings()

        # Підключення сигналу до слоту
        self.processing_stopped.connect(self.on_processing_stopped)

    def on_processing_stopped(self):
        """Оновлення інтерфейсу після зупинки"""
        self.start_stop_btn.setText("▶ Запустити Voice Changer")
        self.status_bar.showMessage("Voice Changer зупинено")
        self.input_device_combo.setEnabled(True)
        self.setup_virtual_devices()
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d28d9, stop:1 #4c1d95);
                border: 2px solid #7c3aed;
                border-radius: 16px;
                padding: 18px;
                font-size: 18px;
                font-weight: bold;
                color: #ffffff;
            }
            QPushButton:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #8b5cf6, stop:1 #7c3aed);
                border: 2px solid #a78bfa;
            }
            QPushButton:pressed {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5b21b6, stop:1 #4c1d95);
                border: 2px solid #6d28d9;
            }
        """)

    def init_ui(self):
        """Ініціалізація користувацького інтерфейсу"""
        self.setWindowTitle("ArchVoice")
        self.setGeometry(100, 100, 1000, 700)

        # Сучасний дизайн 2025: мінімалізм, неон градієнти, прозорість, межі для глибини
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #111827, stop:1 #1f2937);
                color: #f9fafb;
            }
            QTabWidget::pane {
                border: 1px solid #374151;
                background: rgba(31, 41, 55, 220);
                border-radius: 16px;
                margin: 6px;
            }
            QTabBar::tab {
                background: rgba(17, 24, 39, 200);
                color: #d1d5db;
                padding: 14px 28px;
                margin: 6px;
                border-radius: 12px;
                font-weight: 600;
                border: 1px solid #374151;
            }
            QTabBar::tab:selected {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d28d9, stop:1 #5b21b6);
                color: #ffffff;
                border: 2px solid #7c3aed;
            }
            QTabBar::tab:hover {
                background: rgba(55, 65, 81, 220);
                border: 1px solid #4b5563;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 15px;
                border: 1px solid #374151;
                border-radius: 16px;
                margin-top: 1.5ex;
                padding-top: 20px;
                background: rgba(31, 41, 55, 180);
                color: #e5e7eb;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 20px;
                padding: 0 10px;
                color: #9ca3af;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4b5563;
                height: 14px;
                background: #1f2937;
                margin: 5px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d28d9, stop:1 #5b21b6);
                border: 2px solid #4b5563;
                width: 28px;
                margin: -7px 0;
                border-radius: 14px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #8b5cf6, stop:1 #7c3aed);
                border: 2px solid #a78bfa;
            }
            QPushButton {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d28d9, stop:1 #5b21b6);
                border: 2px solid #7c3aed;
                border-radius: 16px;
                padding: 14px 28px;
                font-weight: 600;
                color: #ffffff;
            }
            QPushButton:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #8b5cf6, stop:1 #7c3aed);
                border: 2px solid #a78bfa;
            }
            QPushButton:pressed {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5b21b6, stop:1 #4c1d95);
                border: 2px solid #6d28d9;
            }
            QPushButton:disabled {
                background: #4b5563;
                color: #9ca3af;
                border: 1px solid #374151;
            }
            QListWidget {
                background: rgba(31, 41, 55, 220);
                border: 1px solid #374151;
                border-radius: 16px;
                color: #e5e7eb;
            }
            QListWidget::item {
                padding: 14px;
                border-radius: 12px;
                margin: 6px;
                background: rgba(17, 24, 39, 160);
                border: 1px solid #374151;
            }
            QListWidget::item:selected {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d28d9, stop:1 #5b21b6);
                color: #ffffff;
                border: 2px solid #7c3aed;
            }
            QListWidget::item:hover {
                background: rgba(55, 65, 81, 220);
                border: 1px solid #4b5563;
            }
            QComboBox {
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 12px;
                padding: 10px;
                color: #e5e7eb;
                min-width: 140px;
            }
            QComboBox:hover {
                border: 1px solid #4b5563;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border-left-width: 1px;
                border-left-color: #374151;
                border-left-style: solid;
                border-top-right-radius: 12px;
                border-bottom-right-radius: 12px;
                background: #374151;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 6px solid transparent;
                border-right: 6px solid transparent;
                border-top: 6px solid #e5e7eb;
            }
            QProgressBar {
                border: 1px solid #374151;
                border-radius: 10px;
                text-align: center;
                background: #1f2937;
                color: #e5e7eb;
            }
            QProgressBar::chunk {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #22c55e, stop:1 #16a34a);
                border-radius: 9px;
            }
            QCheckBox {
                color: #e5e7eb;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 24px;
                height: 24px;
                border: 2px solid #4b5563;
                border-radius: 8px;
                background: #1f2937;
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d28d9, stop:1 #5b21b6);
                border: 2px solid #7c3aed;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #a78bfa;
            }
            QTextEdit {
                background: rgba(31, 41, 55, 220);
                border: 1px solid #374151;
                border-radius: 16px;
                color: #e5e7eb;
                padding: 10px;
            }
            QLabel {
                color: #e5e7eb;
            }
            QStatusBar {
                background: rgba(17, 24, 39, 220);
                color: #9ca3af;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("ArchVoice - Змінювач Голосу")
        title.setStyleSheet("""
            font-size: 32px;
            font-weight: 700;
            color: #f9fafb;
            padding: 10px;
        """)

        header_layout.addWidget(title)
        header_layout.addStretch()

        main_layout.addWidget(header)

        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        self.tab_widget = QTabWidget()
        self.tab_widget.setDocumentMode(True)
        main_layout.addWidget(self.tab_widget)

        presets_tab = self.create_presets_tab()
        self.tab_widget.addTab(presets_tab, "🎭 Пресети")

        effects_tab = self.create_effects_tab()
        self.tab_widget.addTab(effects_tab, "🎚️ Налаштування")

        devices_tab = self.create_devices_tab()
        self.tab_widget.addTab(devices_tab, "🎤 Пристрої")

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Готовий до роботи")

    def create_control_panel(self):
        """Створення панелі управління"""
        group = QGroupBox("Управління")
        layout = QHBoxLayout(group)
        layout.setSpacing(24)

        self.start_stop_btn = QPushButton("▶ Запустити Voice Changer")
        self.start_stop_btn.clicked.connect(self.toggle_processing)
        self.start_stop_btn.setMinimumHeight(60)
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #6d28d9, stop:1 #5b21b6);
                border: 2px solid #7c3aed;
                border-radius: 16px;
                padding: 18px;
                font-size: 18px;
                font-weight: 700;
                color: #ffffff;
            }
            QPushButton:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #8b5cf6, stop:1 #7c3aed);
                border: 2px solid #a78bfa;
            }
            QPushButton:pressed {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #5b21b6, stop:1 #4c1d95);
                border: 2px solid #6d28d9;
            }
        """)
        layout.addWidget(self.start_stop_btn)  # Перемістив ліворуч

        level_widget = QWidget()
        level_layout = QVBoxLayout(level_widget)
        level_layout.setSpacing(6)

        level_label = QLabel("Рівень мікрофону:")
        level_label.setStyleSheet("font-weight: 600; color: #9ca3af;")
        level_layout.addWidget(level_label)

        self.level_bar = QProgressBar()
        self.level_bar.setMaximum(100)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(24)
        level_layout.addWidget(self.level_bar)

        layout.addWidget(level_widget)

        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        status_layout.setSpacing(6)

        status_label = QLabel("Статус:")
        status_label.setStyleSheet("font-weight: 600; color: #9ca3af;")
        status_layout.addWidget(status_label)

        self.virtual_mic_status = QLabel("❌ Віртуальний мікрофон: не активний")
        self.virtual_mic_status.setStyleSheet("color: #ef4444;")
        status_layout.addWidget(self.virtual_mic_status)

        layout.addWidget(status_widget)

        return group

    def create_presets_tab(self):
        """Створення вкладки пресетів"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setSpacing(24)
        layout.setContentsMargins(12, 12, 12, 12)

        presets_group = QGroupBox("Готові пресети")
        presets_layout = QVBoxLayout(presets_group)

        self.presets_list = QListWidget()
        self.presets_list.itemClicked.connect(self.load_preset)
        presets_layout.addWidget(self.presets_list)

        self.load_presets_list()

        layout.addWidget(presets_group, 2)

        desc_group = QGroupBox("Опис пресету")
        desc_layout = QVBoxLayout(desc_group)

        self.preset_description = QTextEdit()
        self.preset_description.setReadOnly(True)
        self.preset_description.setMaximumHeight(140)
        desc_layout.addWidget(self.preset_description)

        layout.addWidget(desc_group, 1)

        return widget

    def create_effects_tab(self):
        """Створення вкладки ефектів"""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setSpacing(24)
        layout.setContentsMargins(12, 12, 12, 12)

        main_group = QGroupBox("Основні параметри")
        main_layout = QGridLayout(main_group)
        main_layout.setSpacing(14)

        main_layout.addWidget(QLabel("Висота тону:"), 0, 0)
        self.pitch_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_slider.setRange(50, 200)
        self.pitch_slider.setValue(100)
        self.pitch_slider.valueChanged.connect(self.update_pitch)
        self.pitch_label = QLabel("1.00")
        self.pitch_label.setMinimumWidth(50)
        main_layout.addWidget(self.pitch_slider, 0, 1)
        main_layout.addWidget(self.pitch_label, 0, 2)

        main_layout.addWidget(QLabel("Гучність:"), 1, 0)
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 200)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.volume_label = QLabel("100%")
        self.volume_label.setMinimumWidth(50)
        main_layout.addWidget(self.volume_slider, 1, 1)
        main_layout.addWidget(self.volume_label, 1, 2)

        layout.addWidget(main_group, 0, 0, 1, 2)

        effects_group = QGroupBox("Ефекти")
        effects_layout = QGridLayout(effects_group)
        effects_layout.setSpacing(14)

        effects_layout.addWidget(QLabel("Дисторшн:"), 0, 0)
        self.distortion_slider = QSlider(Qt.Orientation.Horizontal)
        self.distortion_slider.setRange(0, 100)
        self.distortion_slider.setValue(0)
        self.distortion_slider.valueChanged.connect(self.update_distortion)
        self.distortion_label = QLabel("0%")
        self.distortion_label.setMinimumWidth(50)
        effects_layout.addWidget(self.distortion_slider, 0, 1)
        effects_layout.addWidget(self.distortion_label, 0, 2)

        effects_layout.addWidget(QLabel("Ехо (затримка):"), 1, 0)
        self.echo_delay_slider = QSlider(Qt.Orientation.Horizontal)
        self.echo_delay_slider.setRange(0, 500)
        self.echo_delay_slider.setValue(0)
        self.echo_delay_slider.valueChanged.connect(self.update_echo_delay)
        self.echo_delay_label = QLabel("0 мс")
        self.echo_delay_label.setMinimumWidth(50)
        effects_layout.addWidget(self.echo_delay_slider, 1, 1)
        effects_layout.addWidget(self.echo_delay_label, 1, 2)

        effects_layout.addWidget(QLabel("Ехо (відгук):"), 2, 0)
        self.echo_feedback_slider = QSlider(Qt.Orientation.Horizontal)
        self.echo_feedback_slider.setRange(0, 90)
        self.echo_feedback_slider.setValue(0)
        self.echo_feedback_slider.valueChanged.connect(self.update_echo_feedback)
        self.echo_feedback_label = QLabel("0%")
        self.echo_feedback_label.setMinimumWidth(50)
        effects_layout.addWidget(self.echo_feedback_slider, 2, 1)
        effects_layout.addWidget(self.echo_feedback_label, 2, 2)

        effects_layout.addWidget(QLabel("Реверберація:"), 3, 0)
        self.reverb_slider = QSlider(Qt.Orientation.Horizontal)
        self.reverb_slider.setRange(0, 100)
        self.reverb_slider.setValue(0)
        self.reverb_slider.valueChanged.connect(self.update_reverb)
        self.reverb_label = QLabel("0%")
        self.reverb_label.setMinimumWidth(50)
        effects_layout.addWidget(self.reverb_slider, 3, 1)
        self.reverb_label = QLabel("0%")
        effects_layout.addWidget(self.reverb_label, 3, 2)

        self.chorus_checkbox = QCheckBox("Хорус")
        self.chorus_checkbox.toggled.connect(self.update_chorus)
        self.chorus_checkbox.setStyleSheet("""
            QCheckBox {
                color: #e5e7eb;
                font-weight: 600;
            }
        """)
        effects_layout.addWidget(self.chorus_checkbox, 4, 0, 1, 3)

        layout.addWidget(effects_group, 1, 0, 1, 2)

        return widget

    def create_devices_tab(self):
        """Створення вкладки пристроїв"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(24)
        layout.setContentsMargins(12, 12, 12, 12)

        devices_group = QGroupBox("Аудіо пристрої")
        devices_layout = QGridLayout(devices_group)
        devices_layout.setSpacing(14)

        devices_layout.addWidget(QLabel("Справжній мікрофон:"), 0, 0)
        self.input_device_combo = QComboBox()
        devices_layout.addWidget(self.input_device_combo, 0, 1, 1, 2)

        refresh_btn = QPushButton("🔄 Оновити пристрої")
        refresh_btn.clicked.connect(self.setup_audio_devices)
        refresh_btn.setStyleSheet("padding: 10px;")
        devices_layout.addWidget(refresh_btn, 1, 0, 1, 3)

        layout.addWidget(devices_group)

        virtual_group = QGroupBox("Віртуальний мікрофон")
        virtual_layout = QVBoxLayout(virtual_group)
        virtual_layout.setSpacing(14)

        info_label = QLabel("""
        <b>Як використовувати:</b><br>
        1. Натисніть "Створити віртуальний мікрофон"<br>
        2. Запустіть Voice Changer<br>
        3. В інших програмах (Discord, OBS, тощо) оберіть мікрофон "Voice_Changer_Microphone"<br>
        4. Говоріть у свій справжній мікрофон - інші почують оброблений голос!
        """)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #9ca3af; padding: 10px;")
        virtual_layout.addWidget(info_label)

        virtual_controls = QHBoxLayout()
        virtual_controls.setSpacing(14)

        self.create_virtual_btn = QPushButton("➕ Створити віртуальний мікрофон")
        self.create_virtual_btn.clicked.connect(self.create_virtual_microphone)
        self.create_virtual_btn.setStyleSheet("padding: 10px;")
        virtual_controls.addWidget(self.create_virtual_btn)

        self.remove_virtual_btn = QPushButton("➖ Видалити віртуальний мікрофон")
        self.remove_virtual_btn.clicked.connect(self.remove_virtual_microphone)
        self.remove_virtual_btn.setStyleSheet("padding: 10px;")
        virtual_controls.addWidget(self.remove_virtual_btn)

        virtual_layout.addLayout(virtual_controls)
        layout.addWidget(virtual_group)

        advanced_group = QGroupBox("Додаткові налаштування")
        advanced_layout = QGridLayout(advanced_group)
        advanced_layout.setSpacing(14)

        advanced_layout.addWidget(QLabel("Шумоподавлення:"), 0, 0)
        self.noise_gate_slider = QSlider(Qt.Orientation.Horizontal)
        self.noise_gate_slider.setRange(0, 100)
        self.noise_gate_slider.setValue(1)
        self.noise_gate_slider.valueChanged.connect(self.update_noise_gate)
        self.noise_gate_label = QLabel("1%")
        self.noise_gate_label.setMinimumWidth(50)
        advanced_layout.addWidget(self.noise_gate_slider, 0, 1)
        advanced_layout.addWidget(self.noise_gate_label, 0, 2)

        layout.addWidget(advanced_group)
        return widget

    def setup_audio_devices(self):
        """Налаштування списку аудіо пристроїв, відображаючи моно і стерео мікрофони, але з блеклістом"""
        try:
            devices = sd.query_devices()
            blacklist = ['speex', 'stereo', 'monitor', 'virtual', 'jack', 'sof', 'displayport']  # Додано 'stereo' до блекліста

            print("\nПовний список аудіо пристроїв:")
            for i, device in enumerate(devices):
                input_ch = device['max_input_channels']
                output_ch = device['max_output_channels']
                flags = []
                if input_ch > 0:
                    flags.append(f"Мікрофон ({input_ch} канали)")
                if output_ch > 0:
                    flags.append("Вихід")
                flags_str = " | ".join(flags)
                print(f"{i}: {device['name']} [{flags_str}]")

            self.input_device_combo.clear()
            mono_stereo_devices = []
            for i, device in enumerate(devices):
                device_name = device['name'].lower()
                # Дозволяємо пристрої з 1 або 2 каналами, але виключаємо ті, що містять слова з блекліста
                # Виняток: дозволяємо voice_changer_source, навіть якщо він у блеклісті
                if (device['max_input_channels'] in [1, 2] and
                    (not any(keyword in device_name for keyword in blacklist) or
                    'voice_changer_source' in device_name)):
                    self.input_device_combo.addItem(device['name'], i)
                    mono_stereo_devices.append((i, device['name'], device['max_input_channels']))
                    print(f"Додано мікрофон: {device['name']} (ID: {i}, Канали: {device['max_input_channels']})")
                else:
                    reason = ("блекліст" if any(keyword in device_name for keyword in blacklist)
                            else "не 1 або 2 канали")
                    print(f"Пропущено пристрій: {device['name']} (ID: {i}, Канали: {device['max_input_channels']}, Причина: {reason})")

            if not mono_stereo_devices:
                print("Попередження: Не знайдено придатних мікрофонів!")
                QMessageBox.warning(self, "Попередження", "Не знайдено моно або стерео мікрофонів (за винятком блекліста). Перевірте підключені пристрої.")

            # Перевірка наявності віртуального мікрофона
            virtual_input_found = False
            for i, device in enumerate(devices):
                if (device['max_input_channels'] in [1, 2] and
                    'voice_changer_source' in device['name'].lower()):
                    for combo_index in range(self.input_device_combo.count()):
                        if self.input_device_combo.itemData(combo_index) == i:
                            self.input_device_combo.setCurrentIndex(combo_index)
                            virtual_input_found = True
                            print(f"Встановлено віртуальний мікрофон за замовчуванням: {device['name']} (ID: {i})")
                            break
                    break

            # Якщо віртуальний мікрофон не знайдено, встановлюємо дефолтний пристрій
            if not virtual_input_found:
                default_input = sd.default.device[0]
                if default_input is not None:
                    for i in range(self.input_device_combo.count()):
                        if self.input_device_combo.itemData(i) == default_input:
                            device = sd.query_devices(default_input)
                            if (device['max_input_channels'] in [1, 2] and
                                not any(keyword in device['name'].lower() for keyword in blacklist)):
                                self.input_device_combo.setCurrentIndex(i)
                                print(f"Встановлено дефолтний мікрофон: {device['name']} (ID: {default_input}, Канали: {device['max_input_channels']})")
                                break
                    else:
                        print("Попередження: Дефолтний пристрій не відповідає критеріям (не 1 або 2 канали, або в блеклісті)")

            if self.input_device_combo.count() == 0:
                print("Критична помилка: Список придатних мікрофонів порожній!")
                QMessageBox.warning(self, "Помилка", "Не знайдено жодного придатного мікрофона. Перевірте підключення або налаштування звукової системи.")

        except Exception as e:
            print(f"Помилка отримання списку аудіо пристроїв: {e}")
            QMessageBox.warning(self, "Помилка", f"Не вдалося отримати список аудіо пристроїв:\n{e}")

    def setup_virtual_devices(self):
        """Перевірка стану віртуальних пристроїв"""
        try:
            result_sinks = subprocess.run(["pactl", "list", "short", "sinks"],
                                      capture_output=True, text=True)
            result_sources = subprocess.run(["pactl", "list", "short", "sources"],
                                        capture_output=True, text=True)

            sink_exists = any("voice_changer_sink" in line or "Voice_Changer_Output" in line
                            for line in result_sinks.stdout.split('\n'))
            source_exists = any("voice_changer_source" in line or "Voice_Changer_Microphone" in line
                              for line in result_sources.stdout.split('\n'))

            if sink_exists and source_exists:
                self.virtual_mic_status.setText("✅ Віртуальний мікрофон: активний")
                self.virtual_mic_status.setStyleSheet("color: #22c55e;")
                self.create_virtual_btn.setEnabled(False)
                self.remove_virtual_btn.setEnabled(True)
            else:
                self.virtual_mic_status.setText("❌ Віртуальний мікрофон: не активний")
                self.virtual_mic_status.setStyleSheet("color: #ef4444;")
                self.create_virtual_btn.setEnabled(True)
                self.remove_virtual_btn.setEnabled(False)
        except Exception as e:
            print(f"Помилка перевірки віртуальних пристроїв: {e}")
            self.virtual_mic_status.setText("⚠️ Стан віртуального мікрофона: невідомий")
            self.virtual_mic_status.setStyleSheet("color: #f59e0b;")

    def create_virtual_microphone(self):
        """Створення віртуального мікрофону"""
        if self.is_running:
            QMessageBox.warning(self, "Помилка", "Спочатку зупиніть обробку звуку!")
            return

        success, message = self.virtual_mic.create_virtual_devices()
        if success:
            devices = reload_audio_devices()

            print("\nОновлений список пристроїв після створення віртуального мікрофона:")
            for i, dev in enumerate(devices):
                input_ch = dev['max_input_channels']
                output_ch = dev['max_output_channels']
                print(f"{i}: {dev['name']} (in: {input_ch}, out: {output_ch})")

            self.setup_audio_devices()
            self.setup_virtual_devices()

            QMessageBox.information(self, "Успіх",
                message + "\n\nТепер ви можете використовувати 'Voice_Changer_Microphone' в інших програмах!")
        else:
            QMessageBox.critical(self, "Помилка", message)

    def remove_virtual_microphone(self):
        """Видалення віртуального мікрофону"""
        if self.is_running:
            QMessageBox.warning(self, "Увага", "Спочатку зупиніть обробку звуку!")
            return
        self.virtual_mic.remove_virtual_devices()
        QMessageBox.information(self, "Інформація", "Віртуальний мікрофон видалено")
        self.setup_virtual_devices()

    def load_presets_list(self):
        """Завантаження списку пресетів"""
        presets = VoicePresets.get_presets()
        for name, preset in presets.items():
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, preset)
            self.presets_list.addItem(item)

    def load_preset(self, item):
        """Завантаження обраного пресету"""
        preset = item.data(Qt.ItemDataRole.UserRole)
        self.pitch_slider.setValue(int(preset["pitch_shift"] * 100))
        self.volume_slider.setValue(int(preset["volume"] * 100))
        self.distortion_slider.setValue(int(preset["distortion"] * 100))
        self.echo_delay_slider.setValue(int(preset["echo_delay"] * 1000))
        self.echo_feedback_slider.setValue(int(preset["echo_feedback"] * 100))
        self.reverb_slider.setValue(int(preset["reverb_amount"] * 100))
        self.chorus_checkbox.setChecked(preset["chorus_enabled"])
        self.preset_description.setText(f"<b>{item.text()}</b><br><br>{preset['description']}")
        self.status_bar.showMessage(f"Завантажено пресет: {item.text()}")

    def toggle_processing(self):
        """Перемикання обробки аудіо"""
        if not self.is_running:
            # Код для запуску обробки
            try:
                # Перевірка чи віртуальний мікрофон існує
                result_sinks = subprocess.run(["pactl", "list", "short", "sinks"],
                                          capture_output=True, text=True)
                sink_exists = any("voice_changer_sink" in line.lower() or
                                 "voice_changer_output" in line.lower()
                                 for line in result_sinks.stdout.split('\n'))

                if not sink_exists:
                    QMessageBox.warning(self, "Помилка", "Спочатку створіть віртуальний мікрофон!")
                    return

                # Отримуємо ID вибраного пристрою
                input_device = self.input_device_combo.currentData()
                if input_device is None:
                    QMessageBox.warning(self, "Помилка", "Оберіть вхідний мікрофон!")
                    return

                # Виводимо список пристроїв перед запуском
                print("\nПристрої перед запуском обробки:")
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    input_ch = dev['max_input_channels']
                    output_ch = dev['max_output_channels']
                    flags = []
                    if input_ch > 0: flags.append("Мікрофон")
                    if output_ch > 0: flags.append("Вихід")
                    flags_str = " | ".join(flags)
                    print(f"{i}: {dev['name']} [{flags_str}]")

                # Запускаємо обробку
                success, message = self.audio_processor.start_processing(
                    input_device, self.virtual_mic.virtual_sink_name)

                if success:
                    self.is_running = True
                    self.start_stop_btn.setText("⏸ Зупинити Voice Changer")
                    self.status_bar.showMessage("Voice Changer активний - говоріть у мікрофон!")
                    self.input_device_combo.setEnabled(False)
                    self.create_virtual_btn.setEnabled(False)
                    self.remove_virtual_btn.setEnabled(False)
                    # Зміна кольору кнопки на червоний (активний стан)
                    self.start_stop_btn.setStyleSheet("""
                        QPushButton {
                            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                                stop:0 #ef4444, stop:1 #dc2626);
                            border: 2px solid #f87171;
                            border-radius: 16px;
                            padding: 18px;
                            font-size: 18px;
                            font-weight: 700;
                            color: #ffffff;
                        }
                        QPushButton:hover {
                            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                                stop:0 #f87171, stop:1 #ef4444);
                            border: 2px solid #fca5a5;
                        }
                        QPushButton:pressed {
                            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                                stop:0 #dc2626, stop:1 #b91c1c);
                            border: 2px solid #ef4444;
                        }
                    """)
                else:
                    QMessageBox.critical(self, "Помилка", message)
            except Exception as e:
                QMessageBox.critical(self, "Помилка", f"Не вдалося запустити обробку:\n{e}")
        else:
            # Зупинка обробки в окремому потоці
            def stop_processing_thread():
                try:
                    # Зупиняємо обробку
                    self.audio_processor.stop_processing()
                    self.is_running = False

                    # Сигнал для оновлення GUI в головному потоці
                    self.processing_stopped.emit()
                except Exception as e:
                    print(f"Помилка при зупинці: {e}")

            # Запускаємо в окремому потоці
            threading.Thread(target=stop_processing_thread, daemon=True).start()

    def update_level_indicator(self, level):
        """Оновлення індикатора рівня сигналу"""
        level_percent = min(int(level * 1000), 100)
        self.level_bar.setValue(level_percent)

    def update_pitch(self, value):
        """Оновлення висоти тону"""
        pitch = value / 100.0
        self.audio_processor.pitch_shift = pitch
        self.pitch_label.setText(f"{pitch:.2f}")

    def update_volume(self, value):
        """Оновлення гучності"""
        volume = value / 100.0
        self.audio_processor.volume = volume
        self.volume_label.setText(f"{value}%")

    def update_distortion(self, value):
        """Оновлення дисторшну"""
        distortion = value / 100.0
        self.audio_processor.distortion = distortion
        self.distortion_label.setText(f"{value}%")

    def update_echo_delay(self, value):
        """Оновлення затримки ехо"""
        delay = value / 1000.0
        self.audio_processor.echo_delay = delay
        self.echo_delay_label.setText(f"{value} мс")

    def update_echo_feedback(self, value):
        """Оновлення відгуку ехо"""
        feedback = value / 100.0
        self.audio_processor.echo_feedback = feedback
        self.echo_feedback_label.setText(f"{value}%")

    def update_reverb(self, value):
        """Оновлення реверберації"""
        reverb = value / 100.0
        self.audio_processor.reverb_amount = reverb
        self.reverb_label.setText(f"{value}%")

    def update_chorus(self, enabled):
        """Оновлення хорус ефекту"""
        self.audio_processor.chorus_enabled = enabled

    def update_noise_gate(self, value):
        """Оновлення шумоподавлення"""
        threshold = value / 1000.0
        self.audio_processor.noise_gate_threshold = threshold
        self.noise_gate_label.setText(f"{value}%")

    def save_settings(self):
        """Збереження налаштувань"""
        self.settings.setValue("pitch", self.pitch_slider.value())
        self.settings.setValue("volume", self.volume_slider.value())
        self.settings.setValue("distortion", self.distortion_slider.value())
        self.settings.setValue("echo_delay", self.echo_delay_slider.value())
        self.settings.setValue("echo_feedback", self.echo_feedback_slider.value())
        self.settings.setValue("reverb", self.reverb_slider.value())
        self.settings.setValue("chorus", self.chorus_checkbox.isChecked())
        self.settings.setValue("noise_gate", self.noise_gate_slider.value())
        self.settings.setValue("input_device", self.input_device_combo.currentIndex())

    def load_settings(self):
        """Завантаження налаштувань"""
        self.pitch_slider.setValue(self.settings.value("pitch", 100, int))
        self.volume_slider.setValue(self.settings.value("volume", 100, int))
        self.distortion_slider.setValue(self.settings.value("distortion", 0, int))
        self.echo_delay_slider.setValue(self.settings.value("echo_delay", 0, int))
        self.echo_feedback_slider.setValue(self.settings.value("echo_feedback", 0, int))
        self.reverb_slider.setValue(self.settings.value("reverb", 0, int))
        self.chorus_checkbox.setChecked(self.settings.value("chorus", False, bool))
        self.noise_gate_slider.setValue(self.settings.value("noise_gate", 1, int))
        device_index = self.settings.value("input_device", 0, int)
        if device_index < self.input_device_combo.count():
            self.input_device_combo.setCurrentIndex(device_index)

    def closeEvent(self, event):
        """Обробка закриття додатку"""
        if self.is_running:
            self.audio_processor.stop_processing()
        self.save_settings()
        reply = QMessageBox.question(self, 'Закриття додатку',
                                   'Видалити віртуальний мікрофон при закритті?',
                                   QMessageBox.StandardButton.Yes |
                                   QMessageBox.StandardButton.No |
                                   QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Cancel:
            event.ignore()
            return
        elif reply == QMessageBox.StandardButton.Yes:
            self.virtual_mic.remove_virtual_devices()
        event.accept()

def main():
    """Головна функція додатку"""
    app = QApplication(sys.argv)
    app.setApplicationName("ArchVoice")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("ArchVoice")

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
