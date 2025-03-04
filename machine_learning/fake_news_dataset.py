import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Tokenizer for BERT input
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class FakeNewsDataset(Dataset):
    def __init__(self, texts, tabular_data, labels):
        self.texts = texts
        self.tabular_data = tabular_data
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded_text = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        return {
            "input_ids": encoded_text["input_ids"].squeeze(0),  
            "attention_mask": encoded_text["attention_mask"].squeeze(0),
            "tabular_features": self.tabular_data[idx].clone().detach().requires_grad_(True), # depracated: torch.tensor(self.tabular_data[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }