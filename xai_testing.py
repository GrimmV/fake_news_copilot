import pandas as pd
import datasets
import os
import pickle

from model_training import model_testing
from data_preparation import prepare_data
from xai.shap_individual import SHAPIndividual
from config import model_location

# Example usage
if __name__ == "__main__":
    
    # Load training dataset
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])
    
    train_ds = prepare_data(train_raw, name="train")
    
    # Check if cached file exists
    if os.path.exists(model_location):
        with open(model_location, "rb") as f:
            print("Loading cached Model...")
            model = pickle.load(f)
    else:
        print("The model does not exist. Train a model first.")
    
    print(train_ds[0])
    
    model_testing(train_ds, model)
    
    shap_individual = SHAPIndividual(model)
    shap_individual.compute_values(train_ds)