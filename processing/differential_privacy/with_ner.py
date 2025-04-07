import spacy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from opacus import PrivacyEngine
import re

# -----------------------------
# Step 1: NER with spaCy
# -----------------------------
nlp = spacy.load("en_core_web_sm")

def redact_entities(text):
    doc = nlp(text)
    redacted = text
    for ent in doc.ents:
        redacted = redacted.replace(ent.text, f"[{ent.label_}]")
    return redacted

# Sample dataset (PII-containing texts)
texts = [
    "John Smith visited the hospital on January 1st.",
    "Dr. Emily Brown lives in New York.",
    "Contact me at 555-1234 or john@example.com.",
]

# Redact entities
anonymized_texts = [redact_entities(t) for t in texts]
labels = [0, 1, 0]  # Dummy binary labels for training

# -----------------------------
# Step 2: Prepare Dataset
# -----------------------------
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(anonymized_texts).toarray()
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TextDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# -----------------------------
# Step 3: Simple Model
# -----------------------------
model = nn.Sequential(
    nn.Linear(X.shape[1], 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Step 4: Apply Opacus for DP Training
# -----------------------------
privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=0.5,  # Adjust for stronger privacy
    max_grad_norm=0.5,
)

# -----------------------------
# Step 5: Train with DP
# -----------------------------
for epoch in range(5):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -----------------------------
# Step 6: Check ε (privacy budget)
# -----------------------------
epsilon = privacy_engine.get_epsilon(delta=1e-5)
print(f"DP Training completed with ε = {epsilon:.2f}")
