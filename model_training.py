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


def model_training(df, cache_file: str = "model/model.pkl"):

    # Prepare tabular and textual data
    tabular_data = prepare_data(df)

    tabular_data_length = list(tabular_data.size())[1]

    statements = train["statement"].tolist()
    labels = train["label"].tolist()
    
    dataset = FakeNewsDataset(statements, tabular_data, labels)
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=1)
    
        # Check if cached file exists
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print("Loading cached DataFrame...")
            return pickle.load(f), dataset

    # Model initialization
    model = FakeNewsClassifier(num_tabular_features=tabular_data_length)
    
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

    model, processed_dataset = model_training(train)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_ids = processed_dataset[:2]["input_ids"].to(device)
    attention_mask = processed_dataset[:2]["attention_mask"].to(device)
    tabular_features = processed_dataset[:2]["tabular_features"].to(device)
    labels = processed_dataset[:2]["label"].to(device).long()  # Reshape for BCEWithLogitsLoss
    
    sample_output = model.forward(input_ids, attention_mask, tabular_features)
    actual_output = labels
    
    print(f"sample output: {sample_output}")
    print(f"actual output: {actual_output}")
    
    
    
