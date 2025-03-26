import pandas as pd
import datasets

from utils.retrieve_model import retrieve_model
from utils.retrieve_data import DataRetriever

    
if __name__ == "__main__":
    
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])

    X_train = train_raw["statements"].to_list()
    y_train = train_raw["labels"].to_list()

    data_retriever = DataRetriever(X_train)
    
    (
        bow_features,
        bow_feature_names,
        meta_features,
        meta_feature_names,
        combined_features,
        y_train
    ) = data_retriever.generate_input_data(X_train, y_train)
    
    model = retrieve_model()
    
    predictions = model.predict(combined_features)
    probas = model.predict_proba(combined_features)
    proba_df = pd.DataFrame(probas, columns=[f"prob_class_{i}" for i in range(probas.shape[1])])
    
    train_raw["predictions"] = predictions
    train_raw["new_labels"] = y_train
    
    train_raw = pd.concat([train_raw, proba_df], axis=1)
    
    print(train_raw.head())
    train_raw.to_csv("data/basic_train.csv")
    