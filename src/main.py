from src.audio import record_audio_vad
from src.transcription import transcribe_audio
from src.llm import get_ollama_response
from src.tts import text_to_speech_and_play

if __name__ == "__main__":
    conversation_history = []
    system_message = "You are a helpful, concise, and conversational AI assistant. Do not use markdown formatting in your responses."
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