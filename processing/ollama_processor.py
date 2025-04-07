prompt = """You are an anonymization assistant.

Your task is to identify and replace all entities in the following text that belong to any of the following categories, using a consistent identifier format. You must detect and anonymize:

- Persons (PER)
- Locations (LOC) — cities, countries, landmarks, addresses
- Organizations (ORG) — companies, institutions, government bodies
- Nationalities or religious/political groups (NORP)
- Dates (DATE)
- Events (EVENT)
- Products (PRODUCT)
- Laws (LAW)
- Languages (LANGUAGE)
- Works of art (WORK_OF_ART)
- Time values (TIME)
- Percentages (PERCENT)
- Money values (MONEY)
- Quantities (QUANTITY)

Each entity must be replaced with a unique identifier in the format:
- PER_1, PER_2, ...
- LOC_1, LOC_2, ...
- ORG_1, ORG_2, ...
- DATE_1, EVENT_1, etc.

Instructions:
- Replace every full and partial mention of the entity with the correct identifier.
  E.g., "Barack Obama" and "Obama" → PER_1
- Reuse the same identifier consistently across the entire text.
- Do **not** include code or explanations.
- Return **only**:
  1. The processed text with all entities replaced.
  2. A JSON list of all replaced entities with this structure:
     {"<original mention>", "<PII-style ID>"}

Example:

Original: 
"Barack Obama was born in Honolulu in 1961. Obama was elected president in 2008."

Response:
Processed text: 
"[PER_1] was born in [LOC_1] in [DATE_1]. [PER_1] was elected president in [DATE_2]."

JSON:
{
  "Barack Obama": "PER_1",
  "Obama": "PER_1",
  "Honolulu": "LOC_1",
  "1961": "DATE_1",
  "2008": "DATE_2"
}


Now process the following text:

"""

prompt_level_1_template_iterative = """
You are an anonymization assistant.

Task:
Replace the entity '{entity}' (tagged as {identifier}) with a **realistic but random** replacement of the same type. The replacement does **not** need to match the original context exactly, but must sound plausible and belong to the same category.

Category: {category}
Original: {entity}

Output:
Return only the replacement entity, no explanations.
"""

prompt_level_2_template_iterative = """
You are an anonymization assistant.

Task:
Replace the entity '{entity}' (tagged as {identifier}) with a **contextually appropriate** replacement that fits naturally into the surrounding text.

Category: {category}
Context: "{context}"

The replacement should sound natural in this context and match the type of entity ({category}).

Output:
Return only the replacement entity, no explanations.
"""

prompt_level_1_template = """
You are an assistant that replaces placeholders in text with similar values.
Replace with the random word that applies to the same detected cathegory.
Respond with the fully replaced version of the original text (do not include the placeholder version).

Instructions:
- Replace every full and partial mention of the entity with the updated identifier
- Reuse the same identifier consistently across the entire text.
- Do **not** include code or explanations.
- Do not ommit the entities, you must replace all of them.
- Return **only**:
  1. The processed text with all entities replaced.
  2. A JSON list of all replaced entities with this structure:
     {"entity": "<original mention>", "updated_entity": "<updated entity>"}

Example:

Original: 
"Barack Obama was born in Honolulu in 1961. Obama was elected president in 2008."

Response:
Processed text: 
"Joseph Biden was born in Scranton in 1942. Joseph was elected president in 2021."

JSON:
[
  {"entity": "Barack Obama", "updated_entity": "Joseph Biden "},
  {"entity": "Obama", "updated_entity": "Joseph"},
  {"entity": "Honolulu", "updated_entity": "Scranton"},
  {"entity": "1961", "updated_entity": "1942"},
  {"entity": "2008", "updated_entity": "2021"}
]

Now process the following text:

"""


import requests
import re
import json

class LlamaProvider:
    def __init__(self):
        self.url = "http://localhost:11434/api/generate"
        self.model = "llama3.2"

    def call_llama_ollama(self, prompt, model = "llama3.2"):
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
    
    def run(self, text):
        content = prompt + text
        response = self.call_llama_ollama(content)
        print("RESPONSE")
        print(response)
        text_match = re.search(r'Processed text:\s*"(.+?)"\s*JSON:', response, re.DOTALL)
        processed_text = text_match.group(1).strip() if text_match else ""
        print("TEXT MATCH")
        print(text_match)
        print("PROCESSED TEXT")
        print(processed_text)
        json_match = re.search(r'JSON:\s*(\{.*?\}|\[.*?\])', response, re.DOTALL)
        print("JSON MATCH")
        print(json_match)
        json_str = json_match.group(1).strip() if json_match else ""
        print(json_str)
        mappings = json.loads(json_str)
        print("MAPPINGS")
        print(mappings)

        return {
            "masked_text": processed_text,
            "mapping": mappings,
            "raw_response": response
        }
    
    def run_analysis(self, mappings, masked_text):
        counter_present = 0
        counter_not_present = 0
        for key, _ in mappings.items():
            if key in masked_text:
                counter_present +=1
            else:
                counter_not_present +=1
        print("COUNTER PRESENT: ", counter_present, "\nCOUNTER_NOT_PRESENT: ", counter_not_present)
        accuracy = counter_present * 100 / len(mappings.items())
        return accuracy
    
    def apply_masking_level(self, text, mapping, level, context_text):
        if level == 0:
            return text

        elif level == 1:
            prompt = (
                f"{prompt_level_1_template}"
                f"Current text:\n{context_text}\n\n"
                f"Current mapping:\n{mapping}\n\n"
            )
            replaced = self.call_llama_ollama(prompt)
            return replaced.strip()

        elif level == 2:
            prompt = (
                "You are an assistant that replaces placeholders in text with realistic, context-appropriate values.\n"
                "Ensure the replacements match tone, location, gender, and writing style.\n\n"
                f"Original (with placeholders):\n{text}\n\n"
                f"Context:\n{context_text}\n\n"
                f"Current mapping:\n{mapping}\n\n"
                "Respond with the fully replaced version of the original text (do not include the placeholder version)."
            )

            replaced = self.call_llama_ollama(prompt)
            return replaced.strip()

        else:
            raise ValueError("Invalid masking level. Use 0, 1, or 2.")
        
    def apply_masking_level_iteratively(self, mappings, level, context_text, margin=10):

        def find_occurrences_with_margin(text, substring, margin=10):
            results = []
            for match in re.finditer(re.escape(substring), text):
                start, end = match.start(), match.end()
                context_start = max(0, start - margin)
                context_end = min(len(text), end + margin)
                context = text[context_start:context_end]
                results.append({
                    'match': match.group(),
                    'start': start,
                    'end': end,
                    'context': context
                })
            return [r["context"] for r in results]

        new_mapping = {}

        for entity, identifier in mappings.items():
            category = identifier.split('_')[0]

            if level == 1:
                prompt_level_1 = prompt_level_1_template_iterative.format(entity=entity, identifier=identifier, category=category)
                new_replacement = self.call_llama_ollama(prompt_level_1)
                new_mapping[entity] = new_replacement

            elif level == 2:
                context = find_occurrences_with_margin(context_text, entity, margin=margin)
                prompt_level_2 = prompt_level_2_template_iterative.format(entity=entity, identifier=identifier, category=category, context=context)
                new_replacement = self.call_llama_ollama(prompt_level_2)
                new_mapping[entity] = new_replacement

        return new_mapping

    

if __name__=="__main__":
    ollama_provider = LlamaProvider()
    task = """
    Richard Phillips Feynman (May 11, 1918 — February 15, 1988) was an American theoretical physicist, known for his work in the path integral formulation of quantum mechanics as well as his work in particle physics for which he proposed the parton model. His sister was Greta Garbo, born in Vienna, Austria. He worked for Disney Corporation, for Walmart, and for IBM.

    For his contributions to the development of quantum electrodynamics, Feynman received the Nobel Prize in Physics in 1965 jointly with Julian Schwinger and Shin’ichirō Tomonaga. He once had an affair with Cleopatra, the queen of Egypt.

    During his lifetime, Feynman became one of the best-known scientists in the world. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, he was ranked the seventh-greatest physicist of all time along with Albert Einstein and Billy the Kid.

    """
    result = ollama_provider.run(task)
    print("MASKED TEXT")
    masked_text = result["masked_text"]
    print(masked_text)
    mappings = result["mapping"]
    print("MAPPINGS")
    print(mappings)
    accuracy = ollama_provider.run_analysis(mappings, masked_text)
    print("ACCURACY")
    print(accuracy)
    masked_level_1 = ollama_provider.apply_masking_level(masked_text, mappings, 1, task)
    new_mapping = ollama_provider.apply_masking_level_iteratively(mappings, 1, task)
    print("NEW MAPPING LEVEL 1:")
    print(new_mapping)
    new_mapping = ollama_provider.apply_masking_level_iteratively(mappings, 2, task)
    print("NEW MAPPING LEVEL 2:")
    print(new_mapping)
