"""
Модуль графічного інтерфейсу ArchVoice
"""

import sys
import threading
import subprocess
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QSlider, QLabel, QComboBox,
                             QGroupBox, QGridLayout, QCheckBox,
                             QMessageBox, QStatusBar, QProgressBar, QListWidget,
                             QListWidgetItem, QTextEdit, QTabWidget)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtGui import QIcon

from audio_processing import AudioProcessor, VirtualMicrophoneManager
from voice_effects import VoicePresets
from utils import reload_audio_devices


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
        layout.addWidget(self.start_stop_btn)

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
            import sounddevice as sd
            devices = sd.query_devices()
            blacklist = ['speex', 'stereo', 'monitor', 'virtual', 'jack', 'sof', 'displayport']

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
                import sounddevice as sd
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
