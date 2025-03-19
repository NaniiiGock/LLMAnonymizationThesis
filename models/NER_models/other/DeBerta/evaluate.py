# evaluate.py

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig
from model import BertForSequenceTagging
from dataset import NERDataset
from prepare_dataset import build_examples_from_csv
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define label list (must match training)
label_list= [ 'O', 'ACCOUNTNAME', 'ACCOUNTNUMBER', 'AMOUNT', 'BIC', 'BITCOINADDRESS', 'BUILDINGNUMBER', 'CITY', 'COUNTY', 'CREDITCARDCVV', 'CREDITCARDISSUER', 'CREDITCARDNUMBER', 'CURRENCY', 'CURRENCYCODE', 'CURRENCYNAME', 'CURRENCYSYMBOL', 'DISPLAYNAME', 'EMAIL', 'ETHEREUMADDRESS', 'FIRSTNAME', 'FULLNAME', 'GENDER', 'IBAN', 'IP', 'JOBAREA', 'JOBDESCRIPTOR', 'JOBTITLE', 'JOBTYPE', 'LASTNAME', 'LITECOINADDRESS', 'MAC', 'MASKEDNUMBER', 'NAME', 'NEARBYGPSCOORDINATE', 'NUMBER', 'ORDINALDIRECTION', 'PASSWORD', 'PIN', 'SECONDARYADDRESS', 'SEX', 'SEXTYPE', 'STATE', 'STREET', 'STREETADDRESS', 'URL', 'USERAGENT', 'USERNAME', 'ZIPCODE']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(label_list)

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
config = BertConfig.from_pretrained("bert-base-cased", num_labels=num_labels, id2label=id2label, label2id=label2id)

# Load trained model
model = BertForSequenceTagging(config)
model.load_state_dict(torch.load("model.pt"))  # <-- adjust path if needed
model.to(device)
model.eval()

# Load evaluation data
examples = build_examples_from_csv("PII43k.csv", tokenizer, label_list)
_, dev_data = train_test_split(examples, test_size=0.2, random_state=42)

dev_dataset = NERDataset(
    encodings={k: [ex[k] for ex in dev_data] for k in ["input_ids", "attention_mask", "token_type_ids"]},
    labels=[ex["labels"] for ex in dev_data],
    token_starts=[ex["token_starts"] for ex in dev_data]
)
dev_loader = DataLoader(dev_dataset, batch_size=16)

# Evaluation
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in dev_loader:
        input_data = (batch["input_data"][0].to(device), batch["input_data"][1].to(device))
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"]

        outputs = model(input_data=input_data, attention_mask=attention_mask)
        logits = outputs[0]

        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        labels = labels.tolist()

        for pred_seq, label_seq in zip(preds, labels):
            pred_tags = []
            label_tags = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                pred_tags.append(id2label[p])
                label_tags.append(id2label[l])
            all_preds.append(pred_tags)
            all_labels.append(label_tags)

# Print classification report
print(classification_report(all_labels, all_preds, digits=4))
