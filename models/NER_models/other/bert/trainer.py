from datasets import load_from_disk
import json

# Load tokenized dataset
dataset = load_from_disk("data/pii43k_tokenized")
print("dataset loaded...")
# Load label mappings
with open("data/label2id.json") as f:
    label2id = json.load(f)

with open("data/id2label.json") as f:
    id2label = json.load(f)


from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

print("tokenizer and model initialized...")

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="output/pii_ner_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=0.1,
    weight_decay=0.01,
    logging_dir="output/logs",
    logging_steps=10,
)

print("training args set up...")


from transformers import Trainer
from seqeval.metrics import classification_report

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(-1)

    true_preds = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_preds, output_dict=True)
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
print("trainer initialized, starting training...")

trainer.train()
print("training finished...")

model.save_pretrained("output/pii_ner_model")
tokenizer.save_pretrained("output/pii_ner_model")
