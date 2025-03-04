import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Tokenizer for BERT input
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, tabular, labels):
        self.texts = texts
        self.tabular = tabular
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "input_text": self.texts,  
            "tabular": self.tabular,
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }