import pandas as pd
from machine_learning.readability_scorer import ReadabilityScorer
from machine_learning.sentiment_model import SentimentModel
from machine_learning.fake_news_dataset import FakeNewsDataset
import os
import pickle
import pandas as pd
import torch
from config import numerical_cols, use_cached_data

def prepare_data(df: pd.DataFrame, cache_file: str = "processed_data.pkl", name = "train"):
    filename = f"{name}_{cache_file}"
    # Check if cached file exists
    if os.path.exists(filename) and use_cached_data:
        with open(filename, "rb") as f:
            print("Loading cached DataFrame...")
            df = pickle.load(f)
            return _prepare_dataset(df)
    
    print(f"Processing DataFrame '{name}' for the first time...")
    
    readability_scorer = ReadabilityScorer()
    df_features = df["statement"].apply(readability_scorer.analyze_text_complexity).apply(pd.Series)
    df = pd.concat([df, df_features], axis=1)

    sentiment_model = SentimentModel()
    df["sentiment"] = sentiment_model.generate(df["statement"].tolist())
    
    # Save processed DataFrame for future use
    with open(filename, "wb") as f:
        pickle.dump(df, f)

    return _prepare_dataset(df)


def _prepare_dataset(df):
    
    numerical_tensor = torch.tensor(df[numerical_cols].values, dtype=torch.float32)
    statements = df["statement"].tolist()
    labels = df["label"].tolist()
    
    return FakeNewsDataset(statements, numerical_tensor, labels)