import os
import requests
import sounddevice as sd
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import io
import collections
import webrtcvad

def get_ollama_response(messages):
    """
    Sends a request to the local Ollama API and returns the LLM's response.
    Args:
        messages (list): A list of message dictionaries for the conversation.
    Returns:
        str: The LLM's response, or None if an error occurred.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # Default to localhost:11434
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "gemma3:12b",  # Or your desired model
        "messages": messages,  # Use the messages array
        "stream": False
    }

    try:
        response = requests.post(f"{ollama_url}/api/chat", headers=headers, json=data, stream=False)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        try:
            return response.json()["message"]["content"]
        except ValueError:
            print("Error: Response from Ollama is not valid JSON.")
            print("Raw Response Content:")
            print(response.text)  # Print the raw response content
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Ollama API: {e}")
        return None

# New record_audio function with VAD
def record_audio_vad(samplerate=16000, frame_duration_ms=30, vad_aggressiveness=3, silence_limit_ms=1500):
    """
    Records audio from the microphone using VAD to detect speech.
    Returns the recorded audio as a NumPy array.
    """
    vad = webrtcvad.Vad(vad_aggressiveness)
    frames = collections.deque()
    ring_buffer = collections.deque(maxlen=int(silence_limit_ms / frame_duration_ms))
    recorded_audio = []
    speaking = False
    triggered = False

    num_samples_per_frame = int(samplerate * frame_duration_ms / 1000)

    print("Listening for speech...")

    with sd.InputStream(samplerate=samplerate, channels=1, dtype='int16') as stream:
        while True:
            frame_data, overflowed = stream.read(num_samples_per_frame)
            if overflowed:
                print("Warning: Audio buffer overflowed!")

            is_speech = vad.is_speech(frame_data.tobytes(), samplerate)

            if not speaking:
                ring_buffer.append((frame_data, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen: # 90% of frames are speech
                    speaking = True
                    triggered = True
                    print("Speech detected, recording...")
                    for f, s in ring_buffer:
                        recorded_audio.append(f)
                    ring_buffer.clear()
            else:
                recorded_audio.append(frame_data)
                ring_buffer.append((frame_data, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen: # 90% of frames are silence
                    speaking = False
                    print("Silence detected, stopping recording.")
                    break
    
    if not triggered:
        print("No speech detected.")
        return np.array([], dtype=np.int16)

    return np.concatenate(recorded_audio, axis=0)

def transcribe_audio(audio_data, samplerate=16000):
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

    model_size = "turbo"  # Or your desired model size
    model = WhisperModel(model_size, device="cuda", compute_type="int8")
    segments, info = model.transcribe(audio_buffer, beam_size=5)
    detected_language = info.language
    print("Transcription language:", detected_language)
    transcription = "".join(segment.text for segment in segments)
    return transcription, detected_language

def text_to_speech_and_play(text, lang_code=None):
    """
    Sends text to the Kokoro-FastAPI and plays the returned audio.
    Args:
        text (str): The text to convert to speech.
        lang_code (str, optional): The language code for speech generation.
    """
    kokoro_url = "http://localhost:8880/v1/audio/speech"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "kokoro",
        "input": text,
        "voice": "af_heart", # Default voice, can be changed based on language if needed
        "response_format": "wav"
    }

    # Map faster-whisper language codes to Kokoro-FastAPI language codes
    kokoro_lang_map = {
        "en": "a",  # American English
        "es": "e",  # Spanish
        "pt": "p",  # Portuguese (Brazil)
        # Add other mappings as needed based on Kokoro's supported languages
    }

    if lang_code and lang_code in kokoro_lang_map:
        data["lang_code"] = kokoro_lang_map[lang_code]
    elif lang_code:
        print(f"Warning: No Kokoro-FastAPI mapping found for detected language: {lang_code}. Using default voice.")

    try:
        response = requests.post(kokoro_url, headers=headers, json=data)
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

if __name__ == "__main__":
    conversation_history = []
    system_message = "You are a helpful, concise, and conversational AI assistant. Do not use markdown formatting in your responses. Your response has to be in the language the user is talking in."
    conversation_history.append({"role": "system", "content": system_message})

    while True:
        recorded_audio_data = record_audio_vad()
        transcribed_text, detected_language = transcribe_audio(recorded_audio_data)
        print(f"Transcribed Text: {transcribed_text}")
        
        if transcribed_text.lower().strip() in ["exit", "quit"]:
            print("Exiting program.")
            break

        if transcribed_text:
            conversation_history.append({"role": "user", "content": transcribed_text})
            response = get_ollama_response(conversation_history)
            if response:
                print(f"Ollama Response: {response}")
                conversation_history.append({"role": "assistant", "content": response})
                text_to_speech_and_play(response, detected_language)
            else:
                print("Failed to get a response from Ollama.")
        else:
            print("Transcription failed or no speech detected. Please try again.")