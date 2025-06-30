import os
import requests
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import numpy as np

def get_ollama_response(prompt):
    """
    Sends a request to the local Ollama API and returns the LLM's response.
    Args:
        prompt (str): The prompt to send to the LLM.
    Returns:
        str: The LLM's response, or None if an error occurred.
    """
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")  # Default to localhost:11434
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": "gemma3:12b",  # Or your desired model
        "prompt": prompt,  # Use the actual prompt
        "stream": False
    }
    try:
        response = requests.post(f"{ollama_url}/api/generate", headers=headers, json=data, stream=False)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        try:
            return response.json()["response"]
        except ValueError:
            print("Error: Response from Ollama is not valid JSON.")
            print("Raw Response Content:")
            print(response.text)  # Print the raw response content
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Ollama API: {e}")
        return None

def record_audio(file_path, duration=5, samplerate=16000):
    """
    Records audio from the microphone and saves it to a WAV file.
    Args:
        file_path (str): The path to save the WAV file.
        duration (int): The recording duration in seconds.
        samplerate (int): The sample rate.
    """
    print("Recording...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    write(file_path, samplerate, recording)  # Save as WAV file
    print(f"Recording finished and saved to {file_path}")

def transcribe_audio(file_path):
    """
    Transcribes an audio file using faster-whisper.
    Args:
        file_path (str): The path to the audio file.
    Returns:
        str: The transcribed text.
    """
    model_size = "turbo"  # Or your desired model size
    model = WhisperModel(model_size, device="cuda", compute_type="int8")
    segments, info = model.transcribe(file_path, beam_size=5)
    print("Transcription language:", info.language)
    transcription = "".join(segment.text for segment in segments)
    return transcription

if __name__ == "__main__":
    audio_file = "temp_audio.wav"
    record_audio(audio_file)
    transcribed_text = transcribe_audio(audio_file)
    print(f"Transcribed Text: {transcribed_text}")
    
    if transcribed_text:
        response = get_ollama_response(transcribed_text)
        if response:
            print(f"Ollama Response: {response}")
        else:
            print("Failed to get a response from Ollama.")
    else:
        print("Transcription failed.")