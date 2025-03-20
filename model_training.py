import pandas as pd
from machine_learning.model import FakeNewsClassifier
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from data_preparation import prepare_data
import datasets
import torch
import os
import pickle
from config import numerical_cols, use_cached_model, resume_training

import logging

logging.basicConfig(filename='training.log', level=logging.INFO)

def model_training(train_ds, validation_ds, cache_file: str = "model/model.pkl"):
    
    dataloader_train = DataLoader(train_ds, batch_size=50, shuffle=True, num_workers=1)
    dataloader_validation = DataLoader(validation_ds, batch_size=50, shuffle=False, num_workers=1)
    
    # Check if cached file exists
    if os.path.exists(cache_file) and use_cached_model:
        with open(cache_file, "rb") as f:
            print("Loading cached Model...")
            model = pickle.load(f)
            if not resume_training:
                return model            
    
    if not resume_training or not use_cached_model:
        # Model initialization
        model = FakeNewsClassifier(len(numerical_cols))
    
    model.train_model(dataloader_train, dataloader_validation)

    # Save processed DataFrame for future use
    with open(cache_file, "wb") as f:
        pickle.dump(model, f)
        
    return model

def model_testing(test_ds, model):
    
    # Assuming `processed_dataset` is an instance of FakeNewsDataset
    # Create a DataLoader for batching
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Iterate through the DataLoader to extract test data
    for batch in test_loader:
        # Extract inputs and labels
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        tabular_features = batch["tabular"].to(device)  # Move tabular data to device
        labels = batch["label"].to(device).long()  # Move labels to device and convert to long
        
        # Forward pass
        with torch.no_grad():
            sample_output = model(input_ids, attention_mask, tabular_features)

        # Print statements
        print(f"tabular features: {tabular_features.cpu().numpy()}")
        print(f"attention mask: {attention_mask.cpu().numpy()}")
        print(f"sample output (logits): {sample_output.cpu().numpy()}")
        print(f"actual output (labels): {labels.cpu().numpy()}")

        # Break after the first batch for testing
        break   

# Example usage
if __name__ == "__main__":
    
    # Load training dataset
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])
    validation_raw = pd.DataFrame(dataset["validation"])
    test_raw = pd.DataFrame(dataset["test"])
    
    bert_model_name="bert-base-uncased"
    
    train_ds = prepare_data(train_raw, name="train")
    validation_ds = prepare_data(validation_raw, name="validation")
    test_ds = prepare_data(test_raw, name="test")
    
    print(train_ds[0])

    model = model_training(train_ds, validation_ds)
    
    model_testing(test_ds, model)
