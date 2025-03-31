import datasets
import pandas as pd
import json
import pickle

from retrieve_data import DataRetriever
from retrieve_model import retrieve_model
from retrieve_xai import XAIRetriever
from xai.shap_individual import SHAPIndividual


class XAIStorer:
    
    def __init__(self):
        self._curate_initial_data()
        self.model = retrieve_model()
        self.xai_retriever = XAIRetriever()
        
    def store_pdp(self):
        
        partial_dependences = self.xai_retriever.retrieve_partial_dependences()
    
    def store_shap_data(self):
        
        cache

        explainer = self._get_explainer(self.model, self.combined_features[:1000], self.bow_feature_names, self.meta_feature_names)
        explainer.explain(self.combined_features)
        
        tokens = list(map(self.extractor.vectorizer.build_tokenizer(), self.statements))
        combined_feature_names = list(self.bow_feature_names) + list(self.meta_feature_names)
        shap_valuess = explainer.shap_values
        
        explanations = []
        
        for i, shap_values in enumerate(shap_valuess):
            i_tokens = tokens[i]
            i_tokens = [token.lower() for token in i_tokens]
            pred = self.predictions[i]
            elem_id = self.elem_ids[i]
            explanations.append({
                "id": elem_id,
                "prediction": pred,
                "values": {}
            })
            for j, elem in enumerate(shap_values):
                if combined_feature_names[j] in i_tokens or combined_feature_names[j] in self.meta_feature_names:
                    value_key = combined_feature_names[j]
                    explanations[-1]["values"][value_key] = elem[pred]
                
        with open("data/shap.csv", "w") as f:
            json.dump(explanations, f, indent=4)
        
    def _get_explainer(self, model, background_data, bow_feature_names, meta_feature_names, use_cache=True) -> SHAPIndividual:
        
        explainer_path = "model/shap_explainer.pkl"
        
        xai_retriever = XAIRetriever()
        
        if use_cache:
            explainer = xai_retriever.retrieve_shap_explainer()
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
    
    def _curate_initial_data(self):
        dataset = "chengxuphd/liar2"
        dataset = datasets.load_dataset(dataset)
        train_raw = pd.DataFrame(dataset["train"])

        self.statements = train_raw["statement"].to_list()
        self.labels = train_raw["label"].to_list()
        self.ids = train_raw["id"].to_list()

        data_retriever = DataRetriever(self.statements)
        
        # Retrieve trained data and model
        (
            self.bow_features,
            self.bow_feature_names,
            self.meta_features,
            self.meta_feature_names,
            self.combined_features,
            self.labels_simple
        ) = data_retriever.generate_input_data(self.statements, self.labels)
        self.trained_df = data_retriever.retrieve_trained_data()
        self.predictions = self.trained_df['predictions'].to_list()
        self.extractor = data_retriever.extractor