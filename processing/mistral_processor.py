import requests
import re
import json
import random


class MistralProvider:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "mistral"

    def call_mistral_ollama(self, prompt, model = "mistral"):
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


    def mask_pii_with_mistral(self, text):
        system_prompt = (
            "You are an assistant that detects and masks PII in text.\n\n"
            "Given the following input, return the text with entities masked using labels "
            "like [PER_1], [LOC_1], [EMAIL_1], etc., and return a mapping of masked labels "
            "to their original values.\n\n"
            f"Input:\n{text}\n\nOutput:\n"
        )

        response = self.call_mistral_ollama(system_prompt)

        masked_text_match = re.search(r"\s*(.+?)\n\s*Mapping:", response, re.DOTALL)
        mapping_match = re.search(r"Mapping:\s*(.+)", response, re.DOTALL)

        masked_text = masked_text_match.group(1).strip() if masked_text_match else None
        mapping_raw = mapping_match.group(1).strip() if mapping_match else ""

        mapping = {}
        for line in mapping_raw.splitlines():
            parts = line.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                mapping[key] = value

        return {
            "masked_text": masked_text,
            "mapping": mapping,
            "raw_response": response
        }


    def build_function_call_prompt(self, text):
        return (
            "You are a PII anonymization assistant.\n"
            "Given the following input text, return a JSON object with the following fields:\n"
            "- `masked_text`: the original text with PII replaced by tags like [PER_1], [LOC_1], etc.\n"
            "- `mapping`: a dictionary mapping each tag to its original value.\n\n"
            "Respond only with valid JSON.\n\n"
            f"Input:\n{text}\n\n"
            "Output (in JSON):"
        )

    def mask_pii_function_call_style(self, text):
        prompt = self.build_function_call_prompt(text)
        response = self.call_mistral_ollama(prompt)

        try:
            json_start = response.find("{")
            json_data = json.loads(response[json_start:])
        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Failed to parse JSON:\n\n{response}") from e

        return json_data

    def apply_masking_level(self, text: str, mapping: dict, level: int, context_text: str = "") -> str:
        if level == 0:
            return text

        elif level == 1:
            replacements = {
                "PER": ["Alice Smith", "James Lee", "Maria Gonzalez"],
                "LOC": ["Berlin", "London", "Toronto"],
                "ORG": ["OpenAI", "NASA", "UNESCO"],
                "EMAIL": ["user@example.com", "contact@domain.org"],
                "PHONE": ["123-456-7890", "(555) 555-5555"],
            }

            for tag, _ in mapping.items():
                tag_type = tag.strip("[]").split("_")[0]
                if tag_type in replacements:
                    random_value = random.choice(replacements[tag_type])
                    text = text.replace(tag, random_value)
            return text

        elif level == 2:
            prompt = (
                "You are an assistant that replaces placeholders in text with realistic, context-appropriate values.\n"
                "Ensure the replacements match tone, location, gender, and writing style.\n\n"
                f"Original (with placeholders):\n{text}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Current mapping:\n{mapping}\n\n"
                "Respond with the fully replaced version of the original text (do not include the placeholder version)."
            )

            replaced = self.call_mistral_ollama(prompt)
            return replaced.strip()

        else:
            raise ValueError("Invalid masking level. Use 0, 1, or 2.")


    def run(self, text, mode="function_calling"):
        if mode == "function_calling":
            return self.mask_pii_function_call_style(text)
        return self.mask_pii_with_mistral(text)
    
if __name__ == "__main__":

    mistral_provider = MistralProvider()
    long_input_text = """
Hello, my name is Katherine Johnson and I live at 327 Main Street, Palo Alto, California. I was born on August 26, 1985. 
You can reach me via email at katherine.j@example.com or on my mobile at (415) 987-6543.

"""

    input_text = "Hi, I'm Alice Smith from New York. My email is alice@example.com."
    result = mistral_provider.run(input_text, None)
    masked_text = result["masked_text"]
    mapping = result["mapping"]
    print("Masked Text:\n", result["masked_text"])
    print("\nMapping:")
    for tag, value in result["mapping"].items():
        print(f"{tag}: {value}")
    
    final_text = mistral_provider.apply_masking_level(masked_text, mapping, level=2, context_text=input_text)

    print(final_text)