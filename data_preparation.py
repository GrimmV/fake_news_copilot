import pandas as pd
from machine_learning.readability_scorer import ReadabilityScorer
from machine_learning.sentiment_model import SentimentModel
import os
import pickle
import pandas as pd

def prepare_data(df: pd.DataFrame, cache_file: str = "processed_data.pkl", use_cache = True, name = "train"):
    filename = f"{name}_{cache_file}"
    # Check if cached file exists
    if os.path.exists(filename) and use_cache:
        with open(filename, "rb") as f:
            print("Loading cached DataFrame...")
            return pickle.load(f)
    
    print(f"Processing DataFrame '{name}' for the first time...")
    
    readability_scorer = ReadabilityScorer()
    df_features = df["statement"].apply(readability_scorer.analyze_text_complexity).apply(pd.Series)
    df = pd.concat([df, df_features], axis=1)

    sentiment_model = SentimentModel()
    df["sentiment"] = sentiment_model.generate(df["statement"].tolist())
    
    # Save processed DataFrame for future use
    with open(filename, "wb") as f:
        pickle.dump(df, f)

    return df
