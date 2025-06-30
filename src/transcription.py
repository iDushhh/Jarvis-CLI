from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import io
from config import settings

def transcribe_audio(audio_data, samplerate=settings.SAMPLERATE):
    """
    Transcribes audio data using faster-whisper.
    Args:
        audio_data (np.ndarray): The audio data as a NumPy array.
        samplerate (int): The sample rate of the audio data.
    Returns:
        tuple: The transcribed text and the detected language code.
    """
    if audio_data.size == 0:
        return "", ""

    # faster-whisper expects a file path or a file-like object.
    # We can create a BytesIO object from the numpy array.
    # Convert int16 to float32 for faster-whisper
    audio_float32 = audio_data.astype(np.float32) / 32768.0
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_float32, samplerate, format='WAV')
    audio_buffer.seek(0)

    model = WhisperModel(settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE, compute_type=settings.WHISPER_COMPUTE_TYPE)
    segments, info = model.transcribe(audio_buffer, beam_size=5)
    detected_language = info.language
    print("Transcription language:", detected_language)
    transcription = "".join(segment.text for segment in segments)
    return transcription, detected_language