
import re

class PatternProcessor:
    def __init__(self, config):
        self.entity_map = {}

        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b\d{4}[-. ]?\d{4}[-. ]?\d{4}[-. ]?\d{4}\b'
        }

        self.patterns.update(config.get('custom_patterns', {}))

    def preprocess(self, text):
        pattern_replacements = {}
        processed_text = text
        
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                original = match.group(0)
                replacement = f"[{pattern_name.upper()}_{len(self.entity_map)}]"
                processed_text = processed_text.replace(original, replacement)
                self.entity_map[replacement] = original
                pattern_replacements[original] = replacement
        

        return processed_text, pattern_replacements, self.entity_map