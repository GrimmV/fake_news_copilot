import os
import pickle
import pandas as pd
import numpy as np

import datasets
from sklearn.model_selection import train_test_split

from machine_learning.rf.text_feature_extractor import TextFeatureExtractor

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
    
if __name__ == "__main__":
    
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])
    
    model = _retrieve_model()
    
    X_train, X_test, _, _ = train_test_split(train_raw['statement'].to_list(), train_raw['label'].to_list(), test_size=0.0, shuffle=False)
    
    extractor = TextFeatureExtractor(X_train)

    bow_features = extractor.extract_bow_features(X_train)
    meta_features, meta_feature_names = extractor.extract_meta_features(X_train)

    bow_feature_names = extractor.vectorizer.get_feature_names_out()

    # Combine features
    combined_features = np.hstack((bow_features, meta_features))
    
    predictions = model.predict(combined_features)
    
    train_raw["predictions"] = predictions
    
    print(train_raw.head())
    