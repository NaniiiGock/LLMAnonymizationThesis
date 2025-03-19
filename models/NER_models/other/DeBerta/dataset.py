# dataset.py
from torch.utils.data import Dataset
import torch

class NERDataset(Dataset):
    def __init__(self, encodings, labels, token_starts):
        self.encodings = encodings
        self.labels = labels
        self.token_starts = token_starts

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['input_data'] = (item.pop('input_ids'), torch.tensor(self.token_starts[idx]))
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
