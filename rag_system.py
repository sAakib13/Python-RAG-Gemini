import requests

class RAGSystem:
    def __init__(self, api_key):
        self.api_key = "AIzaSyCLUb69RLIbpbTSXlUfQge2FbSrl5UoKXI"
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def generate_response(self, prompt):
        data = {
            "prompt": {
                "text": prompt
            },
            "temperature": 0.7,
            "maxOutputTokens": 256,
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['text']
