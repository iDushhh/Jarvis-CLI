import requests
from config import settings

def get_ollama_response(messages):
    """
    Sends a request to the local Ollama API and returns the LLM's response.
    Args:
        messages (list): A list of message dictionaries for the conversation.
    Returns:
        str: The LLM's response, or None if an error occurred.
    """
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "model": settings.OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(f"{settings.OLLAMA_URL}/api/chat", headers=headers, json=data, stream=False)
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