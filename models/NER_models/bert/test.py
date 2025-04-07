from transformers import pipeline

# Load your model + tokenizer
ner_pipeline = pipeline(
    "ner",
    model="output/pii_ner_model",
    tokenizer="output/pii_ner_model",
    aggregation_strategy="simple"
)

# Test input
text = """
My name is John Smith (employee ID: EMP123456).
Everyone can email me at john.smith@company.com or call 123-456-7890.
I work at Microsoft in Seattle with Sarah Johnson.

My friend's name is Dave Black (employee ID: EMP123445).
Everyone can email him at dave.black@company.com or call 123-456-7890.
He works at Google in Seattle with Florin Dark.
"""

# Predict entities
entities = ner_pipeline(text)

# Function to mask entities in text
def mask_text_by_entity(text, entities):
    # Sort entities by start index (reverse to not shift text while replacing)
    entities = sorted(entities, key=lambda x: x['start'], reverse=True)
    for ent in entities:
        placeholder = f"[{ent['entity_group'].upper()}]"
        text = text[:ent['start']] + placeholder + text[ent['end']:]
    return text

# Apply masking
masked_text = mask_text_by_entity(text, entities)

# Print result
print("🔐 Masked Text:")
print(masked_text)
