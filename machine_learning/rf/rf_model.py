
from sklearn.ensemble import RandomForestClassifier
from machine_learning.rf.text_feature_extractor import TextFeatureExtractor
from sklearn.metrics import accuracy_score
import numpy as np

class RandomForestTextClassifier:
    """Class to train and evaluate a RandomForest classifier for text classification."""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_extractor = TextFeatureExtractor()
    
    def preprocess_data(self, text_data, labels):
        """Preprocess the data by extracting features and combining them."""
        # Extract bag-of-words features
        bow_features = self.feature_extractor.extract_bow_features(text_data)
        
        # Extract meta-information features
        meta_features = self.feature_extractor.extract_meta_features(text_data)
        
        # Combine features
        combined_features = np.hstack((bow_features, meta_features))
        
        return combined_features, labels
    
    def train(self, text_data, labels):
        """Train the RandomForest classifier."""
        X, y = self.preprocess_data(text_data, labels)
        self.clf.fit(X, y)
    
    def evaluate(self, text_data, labels):
        """Evaluate the classifier on test data."""
        X, y = self.preprocess_data(text_data, labels)
        y_pred = self.clf.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy