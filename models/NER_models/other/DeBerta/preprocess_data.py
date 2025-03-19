import pandas as pd
import spacy
from spacy.tokens import DocBin, Span
from tqdm import tqdm
import re

nlp = spacy.blank("en")

def extract_entity_spans(template: str, filled: str):
    """
    Extract entity spans from the mapping between Template and Filled Template.
    Returns list of (start_char, end_char, label)
    """
    entity_spans = []
    pattern = re.compile(r"\[(.*?)\]")
    matches = list(pattern.finditer(template))

    filled_cursor = 0
    temp_cursor = 0

    for match in matches:
        placeholder = match.group(0)
        entity_label = match.group(1).split("_")[0]  # e.g., FULLNAME_1 -> FULLNAME
        before_text = template[temp_cursor:match.start()]
        before_len = len(before_text)

        # Find the actual entity in filled template (rough heuristic)
        filled_cursor = filled.find(before_text, filled_cursor)
        entity_start = filled_cursor + len(before_text)
        next_match_start = matches[matches.index(match)+1].start() if matches.index(match) + 1 < len(matches) else None

        if next_match_start:
            after_text = template[match.end():next_match_start]
        else:
            after_text = template[match.end():]

        if after_text.strip():
            after_pos = filled.find(after_text.strip(), entity_start)
            entity_end = after_pos if after_pos > entity_start else len(filled)
        else:
            entity_end = len(filled)

        # Refine span by trimming whitespace
        entity_text = filled[entity_start:entity_end].strip()
        entity_start = filled.find(entity_text, entity_start)
        entity_end = entity_start + len(entity_text)

        entity_spans.append((entity_start, entity_end, entity_label))
        temp_cursor = match.end()
        filled_cursor = entity_end

    return entity_spans

def create_doc_bin_from_templates(csv_path, output_path):
    df = pd.read_csv(csv_path)
    doc_bin = DocBin(store_user_data=True)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        template = row['Template']
        filled = row['Filled Template']

        doc = nlp.make_doc(filled)
        spans = []

        try:
            entity_offsets = extract_entity_spans(template, filled)
            for start_char, end_char, label in entity_offsets:
                span = doc.char_span(start_char, end_char, label=label, alignment_mode="contract")
                if span is not None:
                    spans.append(span)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

        doc.ents = spacy.util.filter_spans(spans)
        doc_bin.add(doc)

    doc_bin.to_disk(output_path)


from sklearn.model_selection import train_test_split

df = pd.read_csv("PII43k.csv")
train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv("train.csv", index=False)
dev_df.to_csv("dev.csv", index=False)

create_doc_bin_from_templates("train.csv", "data/train.spacy")
create_doc_bin_from_templates("dev.csv", "data/dev.spacy")
