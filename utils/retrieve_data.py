import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List

from utils.text_feature_extractor import TextFeatureExtractor

class DataRetriever:
    
    def __init__(self, X_train):
        
        self.extractor = TextFeatureExtractor(X_train)

    def generate_input_data(self, X: List, y:List, name="train"):
        
        y_new = [self._relabel(elem) for elem in y]

        bow_features = self.extractor.extract_bow_features(X)
        meta_features, meta_feature_names = self.extractor.extract_meta_features(X, name)

        bow_feature_names = self.extractor.vectorizer.get_feature_names_out()

        # Combine features
        combined_features = np.hstack((bow_features, meta_features))

        return (
            bow_features,
            bow_feature_names,
            meta_features,
            meta_feature_names,
            combined_features,
            y_new
        )
        
    
    def retrieve_trained_data(self, file_path="data/basic_train.csv"):
        
        trained_df = pd.read_csv(file_path)
        
        return trained_df

    # Define the mapping function
    def _relabel(self, x):
        if x in [0, 1]:
            return 0
        elif x in [2, 3]:
            return 1
        elif x in [4, 5]:
            return 2