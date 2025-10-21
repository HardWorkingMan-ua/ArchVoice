"""
Модуль для обробки аудіо в реальному часі
"""

import subprocess
import threading
import time
import numpy as np
import sounddevice as sd
from scipy import signal
from PyQt6.QtCore import QObject, pyqtSignal

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
        """Запуск обробки аудіо"""
        try:
            self.is_processing = True
            input_info = sd.query_devices(input_device_id)

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

            return self.start_pipewire_processing(input_device_id, virtual_sink_id)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Помилка запуску: {e}"

    def start_pipewire_processing(self, input_device_id, virtual_sink_id):
        """Спеціальний спосіб запуску для PipeWire"""
        try:
            input_info = sd.query_devices(input_device_id)
            output_info = sd.query_devices(virtual_sink_id)
            self.sample_rate = int(input_info['default_samplerate'])
            input_channels = min(input_info['max_input_channels'], 2)
            output_channels = min(output_info['max_output_channels'], 2)

            print(f"Using sample rate: {self.sample_rate} Hz")
            print(f"Input channels: {input_channels}, Output channels: {output_channels}")

            self.setup_filters()
            self.delay_buffer = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
            self.reverb_buffer = np.zeros(int(self.sample_rate * 0.3), dtype=np.float32)
            self.output_buffer = np.zeros((self.block_size * 100, 2), dtype=np.float32)
            self.buffer_write_index = 0
            self.buffer_read_index = 0

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
            print("Audio streams started")
            return True, "Обробка запущена успішно"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False, f"Помилка запуску: {e}"

    def input_callback(self, indata, frames, time, status):
        """Callback для вхідного потоку"""
        if not self.is_processing:
            return

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
            outdata.fill(0)
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
        if len(audio_data) != self.block_size:
            audio_data = np.pad(audio_data, (0, self.block_size - len(audio_data)), mode='constant')[:self.block_size]

        processed = audio_data.copy()

        # Noise gate
        if np.max(np.abs(processed)) < self.noise_gate_threshold:
            processed = np.zeros_like(processed)

        # High-pass filter
        processed, self.hp_zi = signal.lfilter(self.hp_b, self.hp_a, processed, zi=self.hp_zi)

        # Pitch shift
        processed = self.apply_pitch_shift(processed)

        # Distortion
        if self.distortion > 0:
            processed = self.apply_distortion(processed)

        # Echo
        if self.echo_delay > 0 and self.echo_feedback > 0:
            processed = self.apply_echo(processed)

        # Reverb
        if self.reverb_amount > 0:
            processed = self.apply_reverb(processed)

        # Chorus
        if self.chorus_enabled:
            processed = self.apply_chorus(processed)

        # Low-pass filter
        processed, self.lp_zi = signal.lfilter(self.lp_b, self.lp_a, processed, zi=self.lp_zi)

        # Volume adjustment
        processed *= self.volume

        # Clipping
        processed = np.clip(processed, -0.95, 0.95)

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

        time.sleep(0.1)

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

        with self.buffer_lock:
            if self.output_buffer is not None:
                self.output_buffer.fill(0)
            self.buffer_write_index = 0
            self.buffer_read_index = 0

        print("Обробка повністю зупинена")
