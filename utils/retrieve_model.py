import pickle
import os

def retrieve_model(path="model_rf.pkl"):
    cache = f"model/{path}"
        
    # Check if cached file exists
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            print("Loading cached Model...")
            model = pickle.load(f)
            return model
    else:
        print("No model found. Train a model first, to explain it.")
        return None