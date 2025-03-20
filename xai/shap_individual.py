import shap
import torch
import numpy as np
from transformers import BertTokenizer

class SHAPWrapper:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, texts, structured_features):
        # Tokenize text
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=128
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        # Convert structured features to tensor
        structured_tensor = torch.tensor(structured_features, dtype=torch.float32).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask, structured_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs
    
    
class SHAPIndividual:
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def compute_values(self, texts, structured):
        explainer = shap.KernelExplainer(
            model=SHAPWrapper(self.model, self.tokenizer, self.device),
            data=(texts, structured),  # Background data for estimation
            link="logit"
        )
        
        shap_values = explainer.shap_values(
            (texts, structured),
            nsamples=100  # You can increase for more precise results
        )
        
        print(shap_values)