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
    # Check if cached file exists
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            print("Loading cached DataFrame...")
            return pickle.load(f)

    # Prepare tabular and textual data
    tabular_data = prepare_data(df)

    tabular_data_length = list(tabular_data.size())[1]

    statements = train["statement"].tolist()
    labels = train["label"].tolist()
    
    dataset = FakeNewsDataset(statements, tabular_data, labels)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    # Model initialization
    model = FakeNewsClassifier(num_tabular_features=tabular_data_length)
    
    model.train_model(dataloader)

    # Save processed DataFrame for future use
    with open(cache_file, "wb") as f:
        pickle.dump(model, f)
        
    return model, dataloader


# Example usage
if __name__ == "__main__":
    
    # Load training dataset
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train = pd.DataFrame(dataset["train"])

    model, dataloader = model_training(train)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for batch in dataloader:
        input_ids = batch[0]["input_ids"].to(device)
        attention_mask = batch[0]["attention_mask"].to(device)
        tabular_features = batch[0]["tabular_features"].to(device)
        labels = batch[0]["label"].to(device).unsqueeze(1)  # Reshape for BCEWithLogitsLoss
        break
    
    sample_output = model.forward(input_ids, attention_mask, tabular_features)
    actual_output = labels
    
    print(f"sample output: {sample_output}")
    print(f"actual output: {actual_output}")
    
    
    