import ollama
import json
from collections import defaultdict

MODEL_NAME = "llama3.2"
MAX_STEPS = 5

def llama_anonymize(text, feedback=None, prev_mapping=None):
    feedback_text = f"\nAdvisory Feedback: {feedback}" if feedback else ""
    mapping_text = json.dumps(prev_mapping, indent=2) if prev_mapping else "None"

    prompt = f"""
You are an anonymization assistant trained to redact personal information from text.

Your task is to improve the anonymization of the following text. Use the mapping format [PERSON_1], [DATE_1], etc.

Here is the feedback on what to improve on the previous mapping:
{feedback_text}

Previous Mapping:
{mapping_text}

Input:
"{text}"

Respond with:
Anonymized: <text with tags>
Mapping: <json object mapping each tag to original value>
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
    feedback = None
    mapping = None

    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")

        for i in range(3):
            print("trial" , i)
            result = llama_anonymize(original_text, feedback=feedback, prev_mapping=mapping)
            anonymized = result["anonymized_text"]
            mapping = result["mapping"]
            if mapping is not {}:
                break

        reidentified = llama_adversarial_reidentify(anonymized)

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

        feedback = reidentified

    return best_result

if __name__ == "__main__":
    input_text = """

Марічка, їхня молодша донька, яка ще ходила до школи, була в захваті від цієї подорожі. Вона дуже любила подорожувати і мріяла побачити знамениті пам’ятки Італії, зокрема Пізанську вежу. Її захоплювала ідея зробити безліч фотографій на фоні історичних пам’яток та вуличок італійських міст.

"""
    result = refine_anonymization(input_text)
    print("\n✅ Best anonymization result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
