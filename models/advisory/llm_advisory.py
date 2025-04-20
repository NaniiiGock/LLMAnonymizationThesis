import ollama
import json
from collections import defaultdict

MODEL_NAME = "llama3.2"
MAX_STEPS = 5

def llama_anonymize(text):
    prompt = f"""
You are an anonymization assistant trained to redact personal information from text.

Given a text, replace all sensitive or identifiable entities (like names, dates, organizations, locations, and professions) with numbered tags such as [PERSON_1], [DATE_1], [ORG_1], etc.

After that, return two parts:
1. Anonymized: the text with placeholders.
2. Mapping: a JSON object mapping each placeholder to the original value.

Here is an example:
Input:
"John Smith visited Dr. Green at New York Hospital on March 3rd, 2024."

Response:
Anonymized: "[PERSON_1] visited [PERSON_2] at [ORG_1] on [DATE_1]."
Mapping:
{{
  "[PERSON_1]": "John Smith",
  "[PERSON_2]": "Dr. Green",
  "[ORG_1]": "New York Hospital",
  "[DATE_1]": "March 3rd, 2024"
}}

Now apply the same process to the following input:
"{text}"

Return the result in the same format.
"""
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    content = response["message"]["content"]

    anonymized = ""
    mapping = {}

    if "Anonymized:" in content:
        parts = content.split("Anonymized:")
        remaining = parts[1].strip()

        if "Mapping:" in remaining:
            anonymized_part, mapping_part = remaining.split("Mapping:", 1)
            anonymized = anonymized_part.strip()
            try:
                mapping = json.loads(mapping_part.strip())
            except Exception as e:
                print("[!] Warning: Failed to parse JSON mapping:", e)
        else:
            anonymized = remaining.strip()
    else:
        anonymized = content.strip()

    return {
        "anonymized_text": anonymized,
        "mapping": mapping
    }

def llama_adversarial_reidentify(anonymized_text):
    prompt = f"""
You are an adversarial agent. Try to infer any personal or identifying information from the following anonymized text.

Anonymized Text: "{anonymized_text}"
Inferred Information:
"""
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def refine_anonymization(original_text):
    best_result = None
    best_score = float('inf')

    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")
        result = llama_anonymize(original_text)
        anonymized = result["anonymized_text"]
        mapping = result["mapping"]

        reidentified = llama_adversarial_reidentify(anonymized)

        # Overlap score: count how many original words reappear in adversary response
        original_words = set(original_text.lower().split())
        reidentified_words = set(reidentified.lower().split())
        overlap_score = len(original_words & reidentified_words)

        print("Anonymized:", result)
        print("Reidentified guess:", reidentified)
        print("Overlap Score:", overlap_score)

        if overlap_score < best_score:
            best_score = overlap_score
            best_result = {
                "step": step + 1,
                "anonymized_text": anonymized,
                "mapping": mapping,
                "reidentified_guess": reidentified,
                "score": overlap_score
            }

        if best_score == 0:
            break

    return best_result

if __name__ == "__main__":
    input_text = "John Smith visited Dr. Green at New York Hospital on March 3rd, 2024."
    result = refine_anonymization(input_text)
    print("\n✅ Best anonymization result:")
    print(json.dumps(result, indent=2))
