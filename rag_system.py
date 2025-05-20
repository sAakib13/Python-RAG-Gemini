import requests
import os


class RAGSystem:
    """
    A simple interface for interacting with the Gemini 2.0 Flash API
    to generate responses using a prompt (Retrieval-Augmented Generation).
    """

    def __init__(self, api_key):
        """
        Initialize the RAGSystem with the required Gemini API key.

        Args:
            api_key (str): Your Gemini API key loaded from environment or config.
        """
        self.api_key = api_key
        self.url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.0-flash:generateContent"
        )

    def generate_response(self, prompt):
        """
        Send a prompt to the Gemini API and return the generated response.

        Args:
            prompt (str): The input text prompt, typically including context + user question.

        Returns:
            str: The generated answer text from the Gemini API.

        Raises:
            RuntimeError: If the request to the Gemini API fails.
        """
        try:
            # Prepare query parameters with API key
            params = {"key": self.api_key}

            # Define request payload with the prompt
            payload = {
                "contents": [{
                    "parts": [{"text": f"{prompt}\n\nLimit your answer to 160 characters."}]
                }]
            }

            # Send POST request to Gemini API
            response = requests.post(
                self.url,
                params=params,
                json=payload,
                timeout=30
            )

            # Raise an exception if the response contains an HTTP error
            response.raise_for_status()

            # Extract and return the generated text from the response
            return response.json()['candidates'][0]['content']['parts'][0]['text']

        except Exception as e:
            # Wrap and raise as a RuntimeError with context
            raise RuntimeError(f"Gemini API error: {str(e)}")
