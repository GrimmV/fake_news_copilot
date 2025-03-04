import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Tokenizer for BERT input
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, tabular_df, labels):
        self.texts = texts
        self.tabular_df = tabular_df
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "input_text": torch.tensor(self.texts, dtype=torch.float32),  
            "tabular_df": torch.tensor(self.tabular_df, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }