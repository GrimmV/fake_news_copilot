import shap
from torch.utils.data import DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SHAPIndividual:
    
    def __init__(self, model):
        self.model = model

    def compute_values(self, ds):
        
        explainer = shap.Explainer(self._model_wrapper, masker=shap.maskers.Text(), algorithm="partition")
        
        loader = DataLoader(ds, batch_size=5, shuffle=False, num_workers=1)
        
        for batch in loader:
            
            shap_values = explainer(batch)
            print(shap_values)
            break
            
            

    def _model_wrapper(self, combined_input):
        
        return self.model(combined_input["text"], combined_input["tabular"]).detach().numpy()