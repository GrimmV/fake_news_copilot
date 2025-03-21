import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class FakeNewsDataset(Dataset):
    def __init__(self, texts, tabular, labels, bert_model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        self.texts = texts
        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.tabular = tabular.clone().detach()
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],  
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "tabular": self.tabular[idx],
            "label": self.labels[idx]
        }