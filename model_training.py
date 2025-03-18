import pandas as pd
from machine_learning.model import FakeNewsClassifier
from transformers import BertTokenizer
from machine_learning.fake_news_dataset import FakeNewsDataset
from torch.utils.data import DataLoader
from data_preparation import prepare_data
import datasets
import torch
import os
import pickle
from config import numerical_cols

import logging

logging.basicConfig(filename='training.log', level=logging.INFO)

def model_training(train_df, validation_df, cache_file: str = "model/model.pkl", use_cache = True, resume_training = False):
        
    train_ds = _prepare_dataset(train_df)
    validation_ds = _prepare_dataset(validation_df)
    
    dataloader_train = DataLoader(train_ds, batch_size=50, shuffle=True, num_workers=1)
    dataloader_validation = DataLoader(validation_ds, batch_size=50, shuffle=False, num_workers=1)
    
    # Check if cached file exists
    if os.path.exists(cache_file) and use_cache:
        with open(cache_file, "rb") as f:
            print("Loading cached Dataset...")
            model = pickle.load(f)
            if not resume_training:
                return model, dataset            
    
    if not resume_training:
        # Model initialization
        model = FakeNewsClassifier(len(numerical_cols))
    
    model.train_model(dataloader_train, dataloader_validation)

    # Save processed DataFrame for future use
    with open(cache_file, "wb") as f:
        pickle.dump(model, f)
        
    return model, dataset

def _prepare_dataset(df):
    
    numerical_tensor = torch.tensor(df[numerical_cols].values, dtype=torch.float32)
    statements = df["statement"].tolist()
    labels = df["label"].tolist()
    
    return FakeNewsDataset(statements, numerical_tensor, labels)

# Example usage
if __name__ == "__main__":
    
    # Load training dataset
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])
    validation_raw = pd.DataFrame(dataset["validation"])
    test_raw = pd.DataFrame(dataset["test"])
    
    bert_model_name="bert-base-uncased"
    
    train = prepare_data(train_raw, use_cache=True, name="train")
    validation = prepare_data(validation_raw, use_cache=True, name="validation")
    test = prepare_data(test_raw, use_cache=True, name="test")
    
    print(train.head())
    print(len(train))

    model, processed_dataset = model_training(train, validation, use_cache=True, resume_training=True)
    
    # Assuming `processed_dataset` is an instance of FakeNewsDataset
    # Create a DataLoader for batching
    test_loader = DataLoader(processed_dataset, batch_size=2, shuffle=False, num_workers=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    # Iterate through the DataLoader to extract test data
    for batch in test_loader:
        # Extract inputs and labels
        texts = batch["text"]  # List of tokenized texts (if tokenized) or raw texts
        tabular_features = batch["tabular"].to(device)  # Move tabular data to device
        labels = batch["label"].to(device).long()  # Move labels to device and convert to long
        
        # If texts are not tokenized, tokenize them here
        # Example: Using a tokenizer for transformer models
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        with torch.no_grad():
            sample_output = model(texts, tabular_features)

        # Print statements
        print(f"tabular features: {tabular_features.cpu().numpy()}")
        print(f"attention mask: {attention_mask.cpu().numpy()}")
        print(f"sample output (logits): {sample_output.cpu().numpy()}")
        print(f"actual output (labels): {labels.cpu().numpy()}")

        # Break after the first batch for testing
        break   
