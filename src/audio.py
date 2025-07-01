import sounddevice as sd
import numpy as np
import collections
import webrtcvad
from config import settings
import sys

def print_volume_meter(rms_value, width=50, prefix='Input:  '):
    """Prints a simple text-based volume meter."""
    if np.isnan(rms_value):
        volume = 0
    else:
        # Scale the RMS value to a more reasonable range for visualization
        # This is an arbitrary scaling factor, adjust as needed
        scaled_rms = rms_value / 2000  # Adjust this scaling factor to your liking
        volume = min(int(scaled_rms * width), width)
    bar = '█' * volume + '-' * (width - volume)
    sys.stdout.write(f'\r{prefix}[{bar}]')
    sys.stdout.flush()


def record_audio_vad():
    """
    Records audio from the microphone using VAD to detect speech.
    Returns the recorded audio as a NumPy array.
    """
    vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)
    
    # Buffer for detecting the start of speech
    startup_buffer_duration_ms = 500  # 500ms buffer for startup
    startup_buffer_size = int(startup_buffer_duration_ms / settings.FRAME_DURATION_MS)
    startup_buffer = collections.deque(maxlen=startup_buffer_size)
    
    # Buffer for detecting the end of speech
    silence_buffer_size = int(settings.SILENCE_LIMIT_MS / settings.FRAME_DURATION_MS)
    silence_buffer = collections.deque(maxlen=silence_buffer_size)

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

            rms = np.sqrt(np.mean(frame_data.astype(np.float32)**2))
            is_speech = vad.is_speech(frame_data.tobytes(), settings.SAMPLERATE)

            if not speaking:
                startup_buffer.append((frame_data, is_speech))
                num_voiced = len([f for f, speech in startup_buffer if speech])
                
                if num_voiced > 0.5 * startup_buffer.maxlen: 
                    speaking = True
                    triggered = True
                    sys.stdout.write('Speech detected, recording... ')
                    sys.stdout.flush()
                    for f, s in startup_buffer:
                        recorded_audio.append(f)
                        silence_buffer.append((f,s))
                    startup_buffer.clear()
            else:
                print_volume_meter(rms, prefix="")
                recorded_audio.append(frame_data)
                silence_buffer.append((frame_data, is_speech))
                num_unvoiced = len([f for f, speech in silence_buffer if not speech])
                
                if num_unvoiced > 0.9 * silence_buffer.maxlen: 
                    speaking = False
                    sys.stdout.write('\nSilence detected, stopping recording.\n')
                    sys.stdout.flush()
                    break
    
    if not triggered:
        print("No speech detected.")
        return np.array([], dtype=np.int16)

    return np.concatenate(recorded_audio, axis=0)
