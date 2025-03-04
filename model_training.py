import pandas as pd
from machine_learning.model import FakeNewsClassifier
from machine_learning.fake_news_dataset import FakeNewsDataset
from torch.utils.data import DataLoader
from data_preparation import prepare_data
import datasets
import torch
import os
import pickle

import logging

logging.basicConfig(filename='training.log', level=logging.INFO)


def model_training(df, cache_file: str = "model/model.pkl", use_cache = True):

    numerical_cols = ["Lexical Diversity (TTR)", "Average Word Length", "Avg Syllables per Word", 
                      "Difficult Word Ratio", "Dependency Depth", "Length", "sentiment"]
    categorical_cols = []
    
    numerical_tensor = torch.tensor(df[numerical_cols], dtype=torch.float32)

    statements = train["statement"].tolist()
    labels = train["label"].tolist()
    tabular_data = train[numerical_cols]
    
    dataset = FakeNewsDataset(statements, tabular_data, labels)
    
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=1)
    # Check if cached file exists
    if os.path.exists(cache_file) and use_cache:
        with open(cache_file, "rb") as f:
            print("Loading cached Model...")
            return pickle.load(f), dataset

    # Model initialization
    model = FakeNewsClassifier(len(numerical_tensor))#, num_categories)
    
    model.train_model(dataloader)

    # Save processed DataFrame for future use
    with open(cache_file, "wb") as f:
        pickle.dump(model, f)
        
    return model, dataset


# Example usage
if __name__ == "__main__":
    
    # Load training dataset
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train = pd.DataFrame(dataset["train"])
    
    train = prepare_data(train)

    model, processed_dataset = model_training(train, use_cache=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_ids = processed_dataset[:2]["input_ids"].to(device)
    attention_mask = processed_dataset[:2]["attention_mask"].to(device)
    tabular_features = processed_dataset[:2]["tabular_features"].to(device)
    labels = processed_dataset[:2]["label"].to(device).long()  # Reshape for BCEWithLogitsLoss
    
    sample_output = model.forward(input_ids, attention_mask, tabular_features)
    actual_output = labels
    
    print(f"tabular features: {str(tabular_features)}")
    print(f"statements: {str(attention_mask)}")
    
    print(f"sample output: {sample_output}")
    print(f"actual output: {actual_output}")
    
    
    
