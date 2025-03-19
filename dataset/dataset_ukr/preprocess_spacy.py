import spacy
from spacy.tokens import DocBin

def convert_iob_to_spacy(input_file, output_file, lang="uk"):
    nlp = spacy.blank(lang) 
    doc_bin = DocBin()

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    words, entities = [], []
    start_idx = 0

    for line in lines:
        if line.strip():
            word, tag = line.strip().split()
            words.append(word)

            if tag.startswith("B-"):
                entity_type = tag[2:] 
                start_idx = sum(len(w) + 1 for w in words[:-1])  
                end_idx = start_idx + len(word)
                entities.append((start_idx, end_idx, entity_type))
            elif tag.startswith("I-"):
              
                entities[-1] = (entities[-1][0], sum(len(w) + 1 for w in words), entities[-1][2])
        else:
           
            doc = nlp.make_doc(" ".join(words))
            ents = [doc.char_span(start, end, label=label) for start, end, label in entities]
            ents = [e for e in ents if e is not None]  
            doc.ents = ents
            doc_bin.add(doc)
            words, entities = [], []

    doc_bin.to_disk(output_file)

convert_iob_to_spacy("ner-uk/v2.0/iob/train.iob", "train.spacy")
convert_iob_to_spacy("ner-uk/v2.0/iob/test.iob", "test.spacy")
