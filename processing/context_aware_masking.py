
from ollama_processor import LlamaProvider

class ContexAwareMasker:
    def __init__(self, config):
        self.mode = config.get("mode", "iteratively")
        self.model = config.get("model", "llama")
        self.level = config.get("level", 1)
        self.margin = config.get("margin", 20)

    def mask_iteratively_llama(self, mappings, context_text):
        llama = LlamaProvider()
        new_replacements = llama.apply_masking_level_iteratively(mappings, self.level, context_text, self.margin)
        new_entity_map = {}
        for key, val in new_replacements:
            new_entity_map[val] = key
        return new_replacements, new_entity_map

    def replace_entities_with_masks(self, text, mapping):
        for entity, mask in mapping.items():
            text = text.replace(entity, mask)
        return text
    
    def run(self, mappings, context_text):
        if self.model == "llama":            
            if self.mode == "iteratively":
                return self.mask_iteratively_llama(mappings, self.level, context_text)
            elif self.mode == "one_shot":
                return None
        else:
            return None