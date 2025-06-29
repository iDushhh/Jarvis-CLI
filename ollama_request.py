import os
import requests

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
        "model": "llama2",  # Or your desired model
        "prompt": prompt
    }

    try:
        response = requests.post(f"{ollama_url}/api/generate", headers=headers, json=data, stream=False)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error making request to Ollama API: {e}")
        return None


if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    response = get_ollama_response(user_prompt)

    if response:
        print(f"Ollama Response: {response}")
    else:
        print("Failed to get a response from Ollama.")
