# Jarvis-CLI

Jarvis-CLI is an interactive command-line interface (CLI) assistant that leverages local Large Language Models (LLMs) via Ollama and Text-to-Speech (TTS) capabilities via Kokoro-FastAPI. It allows for natural language interaction through voice input and output, with conversation memory and dynamic language detection.

## Features

*   **Voice Input:** Records audio from your microphone using Voice Activity Detection (VAD) to automatically start and stop recording when you speak.
*   **Speech-to-Text:** Transcribes spoken input to text using `faster-whisper`.
*   **Local LLM Interaction:** Sends transcribed text to a local Ollama instance for natural language processing and response generation.
*   **Conversation Memory:** Maintains context of the conversation, allowing for natural follow-up questions.
*   **Dynamic Language Output:** Detects the language of your input and attempts to generate spoken responses in the same language using Kokoro-FastAPI.
*   **Text-to-Speech:** Converts LLM responses into spoken audio using Kokoro-FastAPI.

## Setup

To get Jarvis-CLI up and running, you'll need to set up your Python environment, Ollama, and Kokoro-FastAPI.

### Prerequisites

*   Python 3.9 or greater
*   `pip` (Python package installer)
*   `git` (for cloning the repository)
*   Docker (for running Ollama and Kokoro-FastAPI)
*   NVIDIA GPU (recommended for `faster-whisper` and Kokoro-FastAPI for optimal performance)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Jarvis-CLI.git # Replace with your repository URL
cd Jarvis-CLI
```

### 2. Set up Python Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

Install the required Python libraries. Note that `webrtcvad` and `sounddevice` might require system-level audio libraries.

```bash
pip install -r requirements.txt
```

**Important for NVIDIA GPU users:**

If you have an NVIDIA GPU, ensure you have the correct CUDA Toolkit and cuDNN libraries installed. `faster-whisper` and Kokoro-FastAPI rely on these. We've included `nvidia-cudnn-cu12` in `requirements.txt` to help, but sometimes manual setup is needed. If you encounter CUDA-related errors, refer to the `faster-whisper` and Kokoro-FastAPI documentation for detailed GPU setup instructions.

### 4. Set up Ollama

Download and run Ollama from its official website: [https://ollama.com/](https://ollama.com/)

Once Ollama is running, pull the `gemma3:12b` model (or your preferred model):

```bash
ollama pull gemma3:12b
```

By default, Jarvis-CLI expects Ollama to be running at `http://localhost:11434`. You can change this in `config/settings.py` if needed.

### 5. Set up Kokoro-FastAPI

Run Kokoro-FastAPI using Docker. It's recommended to use the GPU version if you have an NVIDIA GPU.

**For GPU (NVIDIA):**

```bash
docker run --gpus all -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-gpu
```

**For CPU:**

```bash
docker run -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu
```

By default, Jarvis-CLI expects Kokoro-FastAPI to be running at `http://localhost:8880`. You can change this in `config/settings.py` if needed.

## Usage

Once all prerequisites are met and services are running, activate your virtual environment and run the main script:

```bash
source venv/bin/activate
python src/main.py
```

The CLI will start listening for your voice. Speak your query, and it will transcribe it, send it to Ollama, and speak out the response. To exit the program, simply say "exit" or "quit".

## Project Structure

```
Jarvis-CLI/
├── .git/
├── venv/
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py             # Main entry point, orchestrates the loop
│   ├── audio.py            # Handles microphone input, VAD, recording
│   ├── transcription.py    # Handles faster-whisper transcription
│   ├── llm.py              # Handles Ollama API interaction
│   ├── tts.py              # Handles Kokoro-FastAPI TTS interaction
│   └── utils.py            # (Currently not used, but reserved for future utility functions)
└── config/
    ├── __init__.py
    └── settings.py         # Configuration variables (URLs, model names, VAD params)
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.