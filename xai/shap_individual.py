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
        
        loader = DataLoader(ds, batch_size=50, shuffle=False, num_workers=1)
        
        for batch in loader:
            background = (
                batch["input_ids"].numpy(),
                batch["attention_mask"].numpy(),
                batch["tabular"].numpy()
            )
            break
        
        explainer = shap.DeepExplainer(self.model_predict, background)
        
        shap_values = explainer.shap_values(background)
        print(shap_values)
               
    
    # Define your model predict function
    def model_predict(self, input_ids, attention_mask, tabular_features):
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        tabular_features = torch.tensor(tabular_features)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, tabular_features=tabular_features)
        return outputs