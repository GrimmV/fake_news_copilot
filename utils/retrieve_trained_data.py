import pandas as pd

def retrieve_trained_data(file_path="data/basic_train.csv"):
    
    trained_df = pd.read_csv(file_path)
    
    return trained_df