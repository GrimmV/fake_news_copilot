import shap
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer
from xai.combined_masker import CombinedMasker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SHAPIndividual:
    
    def __init__(self, model):
        self.model = model

    def compute_values(self, ds):
        
        loader = DataLoader(ds, batch_size=100, shuffle=False, num_workers=1)
        
        for batch in loader:
            background = (
                batch["input_ids"].numpy(),
                batch["attention_mask"].numpy(),
                batch["tabular"].numpy()
            )
            break
        
        loader = DataLoader(ds, batch_size=50, shuffle=True, num_workers=1)
        
        for batch in loader:
            sample = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["tabular"]
            )
            break
        
        explainer = shap.DeepExplainer(self.model, background)
        
        shap_values = explainer.shap_values(sample)
        print(shap_values)
               