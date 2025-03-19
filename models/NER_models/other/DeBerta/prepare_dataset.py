import pandas as pd
import re
from transformers import BertTokenizerFast

def extract_entity_spans(template, filled):
    spans = []
    pattern = re.compile(r"\[([A-Z]+)_\d+\]")
    matches = list(pattern.finditer(template))
    cursor_template = 0
    cursor_filled = 0

    for match in matches:
        label = match.group(1)  # e.g., FULLNAME
        before_text = template[cursor_template:match.start()]
        cursor_filled = filled.find(before_text, cursor_filled)
        start_char = cursor_filled + len(before_text)

        cursor_template = match.end()
        after_text = template[match.end():]
        next_placeholder = pattern.search(after_text)
        if next_placeholder:
            after_literal = after_text[:next_placeholder.start()]
            end_char = filled.find(after_literal, start_char)
        else:
            end_char = len(filled)

        entity_text = filled[start_char:end_char].strip()
        start_char = filled.find(entity_text, start_char)
        end_char = start_char + len(entity_text)
        spans.append((start_char, end_char, label))

        cursor_filled = end_char

    return spans

def build_examples_from_csv(csv_path, tokenizer, label_list):
    df = pd.read_csv(csv_path)
    examples = []
    label2id = {label: idx for idx, label in enumerate(label_list)}

    for _, row in df.iterrows():
        filled = str(row['Filled Template'])
        template = str(row['Template'])
        spans = extract_entity_spans(template, filled)

        # Tokenize with word-level mapping
        encoding = tokenizer(filled, return_offsets_mapping=True, truncation=True, padding='max_length')
        offset_mapping = encoding['offset_mapping']
        word_labels = []
        token_starts = []
        prev_word_start = -1

        for idx, (start, end) in enumerate(offset_mapping):
            if start == end:
                word_labels.append(-100)
                token_starts.append(0)
                continue

            assigned = False
            for span_start, span_end, label in spans:
                if span_start <= start < span_end:
                    # prefix = "B" if start == span_start else "I"
                    # word_labels.append(label2id[f"{prefix}-{label}"])
                    word_labels.append(label2id[label])
                    assigned = True
                    break
            if not assigned:
                word_labels.append(label2id["O"])

            # Mark beginning of words
            token_starts.append(1 if start != prev_word_start else 0)
            prev_word_start = start

        examples.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding.get("token_type_ids", None),
            "token_starts": token_starts,
            "labels": word_labels
        })
    return examples
