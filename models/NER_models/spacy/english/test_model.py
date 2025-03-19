import spacy

nlp = spacy.load("NER_models/spacy/english/output/model-best")


text = "In our video conference, discuss the role of evidence in the arbitration process involving Dr. Marvin Rolfson and Julius Daugherty."
doc = nlp(text)

for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
