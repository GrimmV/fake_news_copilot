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

            if len(raw_texts) == 5:
                break

        tabular_tensor = torch.cat(tabular_features, dim=0)  # (5, num_features)

        # Define SHAP explainer with raw text masker
        masker = shap.maskers.Text(tokenizer=self.tokenizer)
        explainer = shap.Explainer(self._model_wrapper(tabular_tensor), masker)

        shap_values = explainer(raw_texts)  # raw_texts is a list of strings
        print(shap_values)

    def _model_wrapper(self, tabular_batch):
        def wrapped_model(raw_texts):
            print(raw_texts)
            # Tokenize raw text into input_ids and attention_mask
            encoded = self.tokenizer(raw_texts, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Move to same device as model if needed
            input_ids = input_ids.to(next(self.model.parameters()).device)
            attention_mask = attention_mask.to(next(self.model.parameters()).device)
            tabular = tabular_batch.to(next(self.model.parameters()).device)

            return self.model(input_ids, attention_mask, tabular).detach().cpu().numpy()

        return wrapped_model
