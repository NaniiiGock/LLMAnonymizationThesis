from providers.llama_provider import OllamaProvider
from providers.mistral_provider import MistralProvider
from providers.openai_provider import OpenAIProvider
from providers.claude_provider import ClaudeProvider
from providers.gemini_provider import GeminiProvider

class GenericProvider:
    def __init__(self, config = {}):
        self.config = config
        self.provider = None
        self.init_provider()

    def init_provider(self):
        providers_mapping = {
            "openai" : OpenAIProvider,
            "llama" : OllamaProvider,
            "mistral": MistralProvider, 
            "gemini": GeminiProvider,
            "claude": ClaudeProvider
        }
        provider = self.config.get("provider", "openai")
        self.provider = providers_mapping[provider](self.config)
    
    async def invoke(self, user_input):
        if isinstance(self.provider, OpenAIProvider):
            prompt = user_input
            response = await self.provider.query_llm(prompt)
        else:
            response = self.provider.query_llm(user_input)
        return response
