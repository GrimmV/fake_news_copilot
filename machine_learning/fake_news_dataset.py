import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
    def __init__(self, texts, tabular, labels):
        self.texts = texts
        self.tabular = tabular
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "encoded_text": list(self.texts[0]),  
            "tabular": self.tabular,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }