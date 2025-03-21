import shap
import torch
import numpy as np
from transformers import BertTokenizer
    
class SHAPIndividual:
    
    def __init__(self, model, background_data, bow_feature_names, meta_feature_names):
        self.explainer = shap.TreeExplainer(model, data=background_data)
        self.bow_feature_names = bow_feature_names
        self.meta_feature_names = meta_feature_names
    
    def explain(self, data):
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(data)
        
        feature_names = np.concatenate((self.bow_feature_names, self.meta_feature_names))
        
        print(shap_values)
        # Visualize SHAP values for the first sample
        print("SHAP Explanation for the first sample:")
        shap.summary_plot(shap_values, data, feature_names=feature_names)