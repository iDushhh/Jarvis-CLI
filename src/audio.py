import sounddevice as sd
import numpy as np
import collections
import webrtcvad
from config import settings

def record_audio_vad():
    """
    Records audio from the microphone using VAD to detect speech.
    Returns the recorded audio as a NumPy array.
    """
    vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
    frames = collections.deque()
    ring_buffer = collections.deque(maxlen=int(settings.SILENCE_LIMIT_MS / settings.FRAME_DURATION_MS))
    recorded_audio = []
    speaking = False
    triggered = False

    num_samples_per_frame = int(settings.SAMPLERATE * settings.FRAME_DURATION_MS / 1000)

    print("Listening for speech...")

    with sd.InputStream(samplerate=settings.SAMPLERATE, channels=1, dtype='int16') as stream:
        while True:
            frame_data, overflowed = stream.read(num_samples_per_frame)
            if overflowed:
                print("Warning: Audio buffer overflowed!")

            is_speech = vad.is_speech(frame_data.tobytes(), settings.SAMPLERATE)

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