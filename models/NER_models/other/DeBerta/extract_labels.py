import pandas as pd
import re

def extract_entity_labels_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    pattern = re.compile(r"\[([A-Z]+)_\d+\]")
    all_labels = set()
    for template in df["Template"]:
        matches = pattern.findall(str(template))
        all_labels.update(matches)
    return sorted(all_labels)

labels = extract_entity_labels_from_csv("PII43k.csv")
print(labels)
