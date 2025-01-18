import requests
import os

class PerplexityAI:
    def __init__(self, api_key: str = None, endpoint: str = "https://api.perplexity.ai/chat/completions"):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.endpoint = endpoint

    def query(self, question: str, model: str = "llama-3.1-sonar-small-128k-online"):
        """Send a query to Perplexity AI and fetch the response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant with access to real-time information."},
                {"role": "user", "content": question}
            ]
        }
        response = requests.post(self.endpoint, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None
