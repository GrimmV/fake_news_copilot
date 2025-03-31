import pandas as pd
import datasets
import json
import pickle

from retrieve_data import DataRetriever
from retrieve_model import retrieve_model
from xai.shap_individual import SHAPIndividual
from retrieve_shap_explainer import retrieve_shap_explainer

def store_shap_data():
    
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])

    X_train = train_raw["statement"].to_list()
    y_train = train_raw["label"].to_list()

    data_retriever = DataRetriever(X_train)
    
    # Retrieve trained data and model
    (
        bow_features,
        bow_feature_names,
        meta_features,
        meta_feature_names,
        combined_features,
        y_train
    ) = data_retriever.generate_input_data(X_train, y_train)
    statements = X_train
    trained_df = data_retriever.retrieve_trained_data()
    predictions = trained_df['predictions'].to_list()
    extractor = data_retriever.extractor

    model = retrieve_model()
    explainer = _get_explainer(model, combined_features[:1000], bow_feature_names, meta_feature_names)
    explainer.explain(combined_features)
    
    tokens = list(map(extractor.vectorizer.build_tokenizer(),statements))
    combined_feature_names = list(bow_feature_names) + list(meta_feature_names)
    shap_valuess = explainer.shap_values
    
    explanations = []
    
    for i, shap_values in enumerate(shap_valuess):
        i_tokens = tokens[i]
        i_tokens = [token.lower() for token in i_tokens]
        pred = predictions[i]
        explanations.append({})
        for j, elem in enumerate(shap_values):
            if combined_feature_names[j] in i_tokens or combined_feature_names[j] in meta_feature_names:
                value_key = combined_feature_names[j]
                explanations[-1][value_key] = elem[pred]
            
    with open("data/shap.csv", "w") as f:
        json.dump(explanations, f, indent=4)
    
def _get_explainer(model, background_data, bow_feature_names, meta_feature_names, use_cache=True) -> SHAPIndividual:
    
    explainer_path = "model/shap_explainer.pkl"
    
    if use_cache:
        explainer = retrieve_shap_explainer()
        if explainer != None:
            return explainer
        
    shap_explainer = SHAPIndividual(
        model,
        background_data=background_data,
        bow_feature_names=bow_feature_names,
        meta_feature_names=meta_feature_names,
    )
    # Save SHAP Explainer for future use
    with open(explainer_path, "wb") as f:
        pickle.dump(shap_explainer, f)
        
    return shap_explainer

if __name__ == "__main__":
    store_shap_data()