import ollama

class OllamaProvider:
    def __init__(self, config={}):
        self.config = config
        self.model = self.config.get("model", "llama3.2")
        self.system_prompt = self.config.get("system_prompt")
        self.temperature = self.config.get("temperature", 0.5)
        self.max_tokens = self.config.get("max_tokens", 1000)
        
    def query_llm(self, prompt):
        
        try:
            response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': prompt
            }
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error querying LLM: {str(e)}"