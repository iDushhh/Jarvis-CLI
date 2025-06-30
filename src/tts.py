import requests
import sounddevice as sd
import soundfile as sf
import io
from config import settings

def text_to_speech_and_play(text, lang_code=None):
    """
    Sends text to the Kokoro-FastAPI and plays the returned audio.
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
        
        print(f"Kokoro-FastAPI Content-Type: {response.headers.get('Content-Type')}")
        print(f"Kokoro-FastAPI Response Content Length: {len(response.content)} bytes")

        try:
            # Read the audio data from the response
            audio_data, samplerate = sf.read(io.BytesIO(response.content))
            
            # Play the audio
            print("Playing response...")
            sd.play(audio_data, samplerate)
            sd.wait()
            print("Finished playing response.")
        except Exception as e:
            print(f"An error occurred while playing the audio: {e}")
            error_file_path = "kokoro_response_error.bin"
            with open(error_file_path, "wb") as f:
                f.write(response.content)
            print(f"Raw Kokoro-FastAPI response content saved to {error_file_path} for inspection.")

    except requests.exceptions.RequestException as e:
        print(f"Error making request to Kokoro-FastAPI: {e}")
        if e.response is not None:
            print(f"Kokoro-FastAPI Response Content: {e.response.text}")