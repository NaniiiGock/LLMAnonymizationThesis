import spacy

nlp = spacy.load("NER_models/spacy/ukrainian/output/model-best")

text = "Іван працює у Києві в компанії Сільпо."
doc = nlp(text)

for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

"""
Entity: Києві, Label: LOC
Entity: Сільпо, Label: ORG
"""

