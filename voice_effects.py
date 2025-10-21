"""
Модуль з пресетами голосу та ефектами
"""

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
