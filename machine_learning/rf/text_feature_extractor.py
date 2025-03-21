import numpy as np
import pandas as pd
import os
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer

from machine_learning.readability_scorer import ReadabilityScorer
from machine_learning.sentiment_model import SentimentModel
from config import use_cached_data

class TextFeatureExtractor:
    """Class to extract features from text data."""
    
    def __init__(self, train_text):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit_transform(train_text).toarray()
    
    def extract_bow_features(self, text_data):
        """Extract bag-of-words features."""
        return self.vectorizer.transform(text_data).toarray()
    
    def extract_meta_features(self, text_data, name = "train"):
        """Extract meta-information features like text length and sentiment."""
        return self._prepare_data(text_data, name=name)
    
    
    def _prepare_data(self, text_data, cache_file: str = "processed_data.pkl", name = "train"):
        filename = f"{name}_{cache_file}"
        # Check if cached file exists
        if os.path.exists(filename) and use_cached_data:
            with open(filename, "rb") as f:
                print("Loading cached DataFrame...")
                df = pickle.load(f)
                print(df.shape)
                return df.to_numpy()
        
        print(f"Processing DataFrame '{name}' for the first time...")
        
        df = pd.DataFrame()
        
        df["statement"] = text_data
        
        readability_scorer = ReadabilityScorer()
        df_features = df["statement"].apply(readability_scorer.analyze_text_complexity).apply(pd.Series)
        df = pd.concat([df, df_features], axis=1)

        sentiment_model = SentimentModel()
        df["sentiment"] = sentiment_model.generate(df["statement"].tolist())
        
        df.drop(columns=["statement"], inplace=True)
        
        print(df.shape)
        # Save processed DataFrame for future use
        with open(filename, "wb") as f:
            print(df.shape)
            pickle.dump(df, f)
        
        return df.to_numpy()