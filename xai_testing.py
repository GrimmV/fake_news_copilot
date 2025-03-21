import pandas as pd
import numpy as np
import os
import pickle
import datasets
from sklearn.model_selection import train_test_split

from machine_learning.rf.text_feature_extractor import TextFeatureExtractor
from xai.shap_individual import SHAPIndividual

def _retrieve_model(path="model_rf.pkl"):
    cache = f"model/{path}"
        
    # Check if cached file exists
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            print("Loading cached Model...")
            model = pickle.load(f)
            return model
    else:
        print("No model found. Train a model first, to explain it.")
        return None

# Example usage
if __name__ == "__main__":
    
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])
    
    X_train, X_test, y_train, y_test = train_test_split(train_raw['statement'].to_list(), train_raw['label'].to_list(), test_size=0.2, random_state=42)
    
    extractor = TextFeatureExtractor(X_train)
    
    bow_features = extractor.extract_bow_features(X_train)
    meta_features, meta_feature_names = extractor.extract_meta_features(X_train)
    
    bow_feature_names = extractor.vectorizer.get_feature_names_out()
    
    # Combine features
    combined_features = np.hstack((bow_features, meta_features))
    
    model = _retrieve_model()
    
    shap_explainer = SHAPIndividual(model, bow_feature_names, meta_feature_names)
    shap_explainer.explain(combined_features)