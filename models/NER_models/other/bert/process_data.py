import pandas as pd
import ast
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from seqeval.metrics import classification_report
import json

# --------------------------
# Step 1: Load and Clean Data
# --------------------------
df = pd.read_csv("data/pii43k.csv")

# Convert stringified lists
df['tokens'] = df['Tokenised Filled Template'].apply(ast.literal_eval)
df['ner_tags'] = df['Tokens'].apply(ast.literal_eval)

# Skip mismatched rows
df = df[df['tokens'].str.len() == df['ner_tags'].str.len()].reset_index(drop=True)

# Encode labels
unique_labels = sorted(set(label for sublist in df['ner_tags'] for label in sublist))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

df['label_ids'] = df['ner_tags'].apply(lambda tags: [label2id[tag] for tag in tags])

# Save mappings
with open("data/label2id.json", "w") as f:
    json.dump(label2id, f)
with open("data/id2label.json", "w") as f:
    json.dump(id2label, f)

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[['tokens', 'label_ids']])
dataset = dataset.train_test_split(test_size=0.2)



# --------------------------
# Step 2: Tokenization + Alignment
# --------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_align_labels(example):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True, padding="max_length", max_length=128)
    word_ids = tokenized.word_ids()
    label_ids = []
    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(example["label_ids"][word_idx])
            previous_word_idx = word_idx
        else:
            label_ids.append(-100)
    tokenized["labels"] = label_ids
    return tokenized

tokenized_dataset = dataset.map(tokenize_and_align_labels, remove_columns=dataset["train"].column_names)


# --------------------------
# Step 3: Model Training
# --------------------------
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="output/pii_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=0.1,
    weight_decay=0.01,
)

def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(-1)
    true_preds = [[id2label.get(p, "O") for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
    true_labels = [[id2label.get(l, "O") for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(preds, labels)]
    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
model.save_pretrained("output/pii_ner_model")
tokenizer.save_pretrained("output/pii_ner_model")

config_path = "output/pii_ner_model/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
config["model_type"] = "bert"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
