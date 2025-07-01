import requests
import sounddevice as sd
import soundfile as sf
import io
import numpy as np
import sys
from config import settings

def print_volume_meter(rms_value, width=50):
    """Prints a simple text-based volume meter."""
    # Scale the RMS value to a more reasonable range for visualization
    # This is an arbitrary scaling factor, adjust as needed
    scaled_rms = rms_value * 10  # Adjust this scaling factor to your liking
    volume = min(int(scaled_rms * width), width)
    bar = '█' * volume + '-' * (width - volume)
    sys.stdout.write(f'\rOutput: [{bar}]')
    sys.stdout.flush()

def text_to_speech_and_play(text, lang_code=None):
    """
    Sends text to the Kokoro-FastAPI and plays the returned audio with a volume meter.
    Args:
        text (str): The text to convert to speech.
        lang_code (str, optional): The language code for speech generation.
    """
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": settings.KOKORO_MODEL,
        "input": text,
        "voice": settings.KOKORO_VOICE,
        "response_format": settings.KOKORO_RESPONSE_FORMAT
    }

    if lang_code and lang_code in settings.KOKORO_LANG_MAP:
        data["lang_code"] = settings.KOKORO_LANG_MAP[lang_code]
    elif lang_code:
        print(f"Warning: No Kokoro-FastAPI mapping found for detected language: {lang_code}. Using default voice.")

    try:
        response = requests.post(settings.KOKORO_URL, headers=headers, json=data)
        response.raise_for_status()

        try:
            audio_data, samplerate = sf.read(io.BytesIO(response.content))
            
            # Ensure audio is in a playable format (e.g., float32)
            if audio_data.dtype != 'float32':
                audio_data = audio_data.astype(np.float32)

            print("Playing response...")
            with sd.OutputStream(samplerate=samplerate, channels=audio_data.shape[1] if audio_data.ndim > 1 else 1, dtype='float32') as stream:
                block_size = 1024
                for i in range(0, len(audio_data), block_size):
                    block = audio_data[i:i+block_size]
                    stream.write(block)
                    rms = np.sqrt(np.mean(block**2))
                    print_volume_meter(rms)
            sys.stdout.write('\n')
            sys.stdout.flush()
            print("Finished playing response.")
        except Exception as e:
            print(f"An error occurred while playing the audio: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error making request to Kokoro-FastAPI: {e}")