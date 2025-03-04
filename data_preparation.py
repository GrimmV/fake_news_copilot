import pandas as pd
from machine_learning.readability_scorer import ReadabilityScorer
from machine_learning.sentiment_model import SentimentModel
from machine_learning.preprocessor import Preprocessor
import os
import pickle
import pandas as pd

def prepare_data(df: pd.DataFrame, cache_file: str = "processed_data.pkl", use_cache = True):
    # Check if cached file exists
    if os.path.exists(cache_file) and use_cache:
        with open(cache_file, "rb") as f:
            print("Loading cached DataFrame...")
            return pickle.load(f)
    
    print("Processing DataFrame for the first time...")
    train = df.copy()
    
    readability_scorer = ReadabilityScorer()
    df_features = train["statement"].apply(readability_scorer.analyze_text_complexity).apply(pd.Series)
    train = pd.concat([train, df_features], axis=1)

    sentiment_model = SentimentModel()
    train["sentiment"] = sentiment_model.generate(train["statement"].tolist())
    
    # Save processed DataFrame for future use
    with open(cache_file, "wb") as f:
        pickle.dump(train, f)

    return train
