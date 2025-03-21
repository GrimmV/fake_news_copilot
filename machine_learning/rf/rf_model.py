
from sklearn.ensemble import RandomForestClassifier
from machine_learning.rf.text_feature_extractor import TextFeatureExtractor
from sklearn.metrics import accuracy_score
import numpy as np
import os
import pickle

from config import use_cached_model

class RandomForestTextClassifier:
    """Class to train and evaluate a RandomForest classifier for text classification."""
    
    def __init__(self, train_text, n_estimators=100, random_state=42, max_depth=80):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
        self.feature_extractor = TextFeatureExtractor(train_text)
    
    def preprocess_data(self, text_data, labels, name = "train"):
        """Preprocess the data by extracting features and combining them."""
        # Extract bag-of-words features
        bow_features = self.feature_extractor.extract_bow_features(text_data)
        
        # Extract meta-information features
        meta_features = self.feature_extractor.extract_meta_features(text_data, name=name)
        
        print(bow_features.shape)
        print(meta_features.shape)
        
        # Combine features
        combined_features = np.hstack((bow_features, meta_features))
        
        return combined_features, labels
    
    def train(self, text_data, labels, cache_file: str = "model_rf.pkl"):
        
        cache = f"model/{cache_file}"
        
        # Check if cached file exists
        if os.path.exists(cache) and use_cached_model:
            with open(cache, "rb") as f:
                print("Loading cached Model...")
                model = pickle.load(f)
                self.clf = model
        else:
            """Train the RandomForest classifier."""
            X, y = self.preprocess_data(text_data, labels)
        
            
            self.clf.fit(X, y)
            # Save processed DataFrame for future use
            with open(cache, "wb") as f:
                pickle.dump(self.clf, f)
        

    
    def evaluate(self, text_data, labels, name: str):
        """Evaluate the classifier on test data."""
        X, y = self.preprocess_data(text_data, labels, name)
        y_pred = self.clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy