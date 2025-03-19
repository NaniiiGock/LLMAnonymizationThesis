import ollama

class OllamaProvider:
    def __init__(self, config):
        self.model = config["model"]
        self.system_prompt = config["system_prompt"]
        self.temperature = config["temperature"]
        self.max_tokens = config["max_tokens"]
        
    def query_llm(self, prompt, task):
        
        try:
            response = ollama.chat(model='llama3.2', messages=[
            {
                'role': 'user',
                'content': prompt + task
            }
            ])

            print(response['message']['content'])
        except Exception as e:
            return f"Error querying LLM: {str(e)}"