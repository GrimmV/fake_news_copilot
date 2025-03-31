import pickle
import os

def retrieve_shap_explainer(path="shap_explainer.pkl"):
    cache = f"model/{path}"
        
    # Check if cached file exists
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            print("Loading cached Shap Explainer...")
            model = pickle.load(f)
            return model
    else:
        print("No explainer found. Create shap explainer first.")
        return None