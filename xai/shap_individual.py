import shap
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
from combined_masker import CombinedMasker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SHAPIndividual:
    
    def __init__(self, model, bert_model_name="bert-base-uncased"):
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def compute_values(self, ds):
        
        combined_masker = self._masking()
        
        explainer = shap.Explainer(self._model_wrapper, masker=combined_masker)
        
        loader = DataLoader(ds, batch_size=5, shuffle=False, num_workers=1)
        
        # Iterate through the DataLoader to extract test data
        for batch in loader:
            input_sample = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["tabular"]
            )
            
            shap_values = explainer(input_sample)
            print(shap_values)
            break
            

    def _model_wrapper(self, input_sample):
        input_ids, attention_mask, tabular = input_sample
        
        return self.model(input_ids, attention_mask, tabular).detach().numpy()
    
    def _masking(self):
        
        # Define the custom masker for both text and structured inputs
        text_masker = shap.maskers.Text()  # Masker for text inputs
        structured_masker = shap.maskers.Independent(masker=shap.maskers.Tuples([]))  # Independent mask for structured
        
        return CombinedMasker(text_masker, structured_masker)