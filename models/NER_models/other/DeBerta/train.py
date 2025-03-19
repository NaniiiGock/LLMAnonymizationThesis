# train.py
from transformers import BertTokenizerFast, BertConfig
from torch.utils.data import DataLoader
import torch
from model import BertForSequenceTagging
from dataset import NERDataset
from prepare_dataset import build_examples_from_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_list = ['O', 'B-FULLNAME', 'I-FULLNAME', 'B-CITY', 'I-CITY', 'B-STATE', 'I-STATE', 'B-NAME', 'I-NAME']
label_list= ['O','ACCOUNTNAME', 'ACCOUNTNUMBER', 'AMOUNT', 'BIC', 'BITCOINADDRESS', 'BUILDINGNUMBER', 'CITY', 'COUNTY', 'CREDITCARDCVV', 'CREDITCARDISSUER', 'CREDITCARDNUMBER', 'CURRENCY', 'CURRENCYCODE', 'CURRENCYNAME', 'CURRENCYSYMBOL', 'DISPLAYNAME', 'EMAIL', 'ETHEREUMADDRESS', 'FIRSTNAME', 'FULLNAME', 'GENDER', 'IBAN', 'IP', 'JOBAREA', 'JOBDESCRIPTOR', 'JOBTITLE', 'JOBTYPE', 'LASTNAME', 'LITECOINADDRESS', 'MAC', 'MASKEDNUMBER', 'NAME', 'NEARBYGPSCOORDINATE', 'NUMBER', 'ORDINALDIRECTION', 'PASSWORD', 'PIN', 'SECONDARYADDRESS', 'SEX', 'SEXTYPE', 'STATE', 'STREET', 'STREETADDRESS', 'URL', 'USERAGENT', 'USERNAME', 'ZIPCODE']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}
num_labels = len(label_list)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
config = BertConfig.from_pretrained("bert-base-cased", num_labels=num_labels, id2label=id2label, label2id=label2id)
model = BertForSequenceTagging(config).to(device)

# Read dataset
examples = build_examples_from_csv("PII43k.csv", tokenizer, label_list)

# Split train/dev manually (simple)
split = int(len(examples) * 0.8)
train_data = examples[:split]
dev_data = examples[split:]

train_dataset = NERDataset(
    encodings={k: [ex[k] for ex in train_data] for k in ["input_ids", "attention_mask", "token_type_ids"]},
    labels=[ex["labels"] for ex in train_data],
    token_starts=[ex["token_starts"] for ex in train_data]
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):
    for batch in train_loader:
        input_data = (batch["input_data"][0].to(device), batch["input_data"][1].to(device))
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_data=input_data, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "model.pt")
