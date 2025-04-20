import requests

class MistralProvider:
    def __init__(self, config={}):
        self.config = config
        self.url = "http://localhost:11434/api/generate"
        self.model = "mistral"

    def query_llm(self, prompt, model = "mistral"):
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload)
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.status_code} - {response.text}")
        return response.json()["response"]
