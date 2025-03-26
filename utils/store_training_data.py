import pandas as pd

from utils.retrieve_model import retrieve_model
from utils.retrieve_data import retrieve_data
    
if __name__ == "__main__":
    
    (
        bow_features,
        bow_feature_names,
        meta_features,
        meta_feature_names,
        combined_features,
        y_train,
        train_raw
    ) = retrieve_data()
    
    model = retrieve_model()
    
    predictions = model.predict(combined_features)
    probas = model.predict_proba(combined_features)
    proba_df = pd.DataFrame(probas, columns=[f"prob_class_{i}" for i in range(probas.shape[1])])
    
    train_raw["predictions"] = predictions
    train_raw["new_labels"] = y_train
    
    train_raw = pd.concat([train_raw, proba_df], axis=1)
    
    print(train_raw.head())
    train_raw.to_csv("data/basic_train.csv")
    