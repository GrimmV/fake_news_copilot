import torch
from torch.utils.data import Dataset

class FakeNewsDataset(Dataset):
    def __init__(self, texts, tabular, labels):
        self.texts = texts
        self.tabular = tabular.clone().detach()
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],  
            "tabular": self.tabular[idx],
            "label": self.labels[idx]
        }