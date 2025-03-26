
from sklearn.ensemble import RandomForestClassifier
from utils.text_feature_extractor import TextFeatureExtractor
from sklearn.metrics import accuracy_score
import numpy as np
import os
import pickle

from config import use_cached_model

class RandomForestTextClassifier:
    """Class to train and evaluate a RandomForest classifier for text classification."""
    
    def __init__(self, train_text, n_estimators=100, random_state=42, max_depth=60):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    
    def train(self, X, y, cache_file: str = "model_rf.pkl"):
        
        cache = f"model/{cache_file}"
        
        # Check if cached file exists
        if os.path.exists(cache) and use_cached_model:
            with open(cache, "rb") as f:
                print("Loading cached Model...")
                model = pickle.load(f)
                self.clf = model
        else:
            """Train the RandomForest classifier."""
            self.clf.fit(X, y)
            # Save processed DataFrame for future use
            with open(cache, "wb") as f:
                pickle.dump(self.clf, f)
        

    
    def evaluate(self, X, y):
        """Evaluate the classifier on test data."""
        y_pred = self.clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy