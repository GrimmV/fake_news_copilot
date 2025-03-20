import shap
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SHAPIndividual:
    
    def __init__(self, model):
        self.model = model
        self.tokenizer = None

    def compute_values(self, ds):
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
        self.tokenizer = ds.tokenizer

        # Collect raw text and tabular features for SHAP
        raw_texts = []
        tabular_features = []

        for batch in loader:
            raw_texts.append(batch["text"][0])  # batch_size=1
            tabular_features.append(batch["tabular"][0].unsqueeze(0))  # Shape: (1, num_features)

            if len(raw_texts) == 1:
                break

        tabular_tensor = torch.cat(tabular_features, dim=0)  # (5, num_features)

        # Define SHAP explainer with raw text masker
        masker = shap.maskers.Text(tokenizer=self.tokenizer)
        explainer = shap.Explainer(self._model_wrapper(tabular_tensor), masker)
        
        print("This is the text: \n")
        print(raw_texts)

        shap_values = explainer(raw_texts)  # raw_texts is a list of strings
        print(shap_values)

    def _model_wrapper(self, tabular_batch):
        def wrapped_model(raw_texts):
            # Tokenize raw text into input_ids and attention_mask
            encoded = self.tokenizer(list(raw_texts), padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Move to same device as model if needed
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            tabular = tabular_batch.to(device)
            
            print(f"input_ids shape: {input_ids.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"tabular shape: {tabular.shape}")

            # Pass through the model
            outputs = self.model(input_ids, attention_mask, tabular)
            return outputs.detach().cpu().numpy()

        return wrapped_model