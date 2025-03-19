import spacy
from spacy.tokens import DocBin
from spacy.training.example import Example

nlp = spacy.blank("en")
db = DocBin()

TRAIN_DATA = [
    ("DeBERTa is developed by Microsoft.", {"entities": [(0, 7, "MODEL"), (26, 35, "ORG")]}),
    ("OpenAI created ChatGPT.", {"entities": [(0, 6, "ORG"), (15, 22, "PRODUCT")]}),
]

for text, ann in TRAIN_DATA:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in ann["entities"]:
        span = doc.char_span(start, end, label=label)
        if span:
            ents.append(span)
    doc.ents = ents
    db.add(doc)

db.to_disk("train.spacy")
