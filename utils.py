"""
Допоміжні функції для роботи з аудіо
"""

import sounddevice as sd

def reload_audio_devices():
    """Перезавантаження звукової системи для оновлення списку пристроїв"""
    try:
        sd._terminate()
        sd._initialize()
        return sd.query_devices()
    except Exception as e:
        print(f"Помилка перезавантаження звукової системи: {e}")
        return sd.query_devices()
