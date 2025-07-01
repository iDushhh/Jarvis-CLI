import os

# Ollama API settings
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = "gemma3:12b"

# Kokoro-FastAPI settings
KOKORO_URL = "http://localhost:8880/v1/audio/speech"
KOKORO_MODEL = "kokoro"
KOKORO_VOICE = "af_heart"
KOKORO_RESPONSE_FORMAT = "wav"

# Audio recording settings
SAMPLERATE = 16000
FRAME_DURATION_MS = 30
VAD_AGGRESSIVENESS = 3
SILENCE_LIMIT_MS = 3000

# Faster-Whisper settings
WHISPER_MODEL_SIZE = "turbo"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# Language mapping for Kokoro-FastAPI
KOKORO_LANG_MAP = {
    "en": "a",  # American English
    "es": "e",  # Spanish
    "pt": "p",  # Portuguese (Brazil)
    # Add other mappings as needed based on Kokoro's supported languages
}
