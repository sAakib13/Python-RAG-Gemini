import requests
import os


class RAGSystem:
    def __init__(self, api_key):
        self.api_key = api_key
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    def generate_response(self, prompt):
        try:
            params = {"key": self.api_key}
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }

            response = requests.post(
                self.url,
                params=params,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            return response.json()['candidates'][0]['content']['parts'][0]['text']

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")
