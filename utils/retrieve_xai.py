
import pickle
import os
import json
import pandas as pd
import datasets

from sklearn.inspection import partial_dependence

from utils.xai.shap_individual import SHAPIndividual
from utils.xai.proximity_based.similarity_handler import SimilarityHandler
from utils.xai.proximity_based.similars import SimilarPredHandler
from utils.xai.proximity_based.counterfactuals import CounterfactualHandler
from utils.retrieve_data import DataRetriever
from utils.retrieve_model import retrieve_model

class XAIRetriever:
    
    def __init__(self):
        self._curate_initial_data()
        self.model = retrieve_model()
        self.similarity_handler = SimilarityHandler(self.trained_df.to_dict("records"))
        
    def retrieve_similars(self, use_cache=True):
        
        cache = "data/similars.csv"
        
        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached similars...")
                shap_explainer = json.load(f)
                return shap_explainer
            
        similar_pred_handler = SimilarPredHandler(self.similarity_handler)
        similars = []
        for i, statement in enumerate(self.statements):
            similar_datapoints = similar_pred_handler.find_similars(statement, self.predictions[i])
            similars.append({
                "id": self.ids[i],
                "similars": similar_datapoints
            })
            print(self.ids[i])
            
        with open(cache, "w") as f:
            json.dump(similars, f, indent=4)
            
        return similars


    def retrieve_counterfactuals(self, use_cache=True):
        
        cache = "data/counterfactuals.csv"
        
        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached counterfactuals...")
                counterfactuals = json.load(f)
                return counterfactuals
            
        counterfactuals_handler = CounterfactualHandler(self.similarity_handler)
        counterfactuals = []
        for i, statement in enumerate(self.statements):
            counterfactual_datapoints = counterfactuals_handler.find_counterfactuals(statement, self.predictions[i])
            counterfactuals.append({
                "id": self.ids[i],
                "counterfactuals": counterfactual_datapoints
            })
            print(self.ids[i])
            
        with open(cache, "w") as f:
            json.dump(counterfactuals, f, indent=4)
            
        return counterfactuals
    
    def retrieve_shap_explainer(self, use_cache=True) -> SHAPIndividual:
        
        cache = "model/shap_explainer.pkl"
        
        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached shap explainer...")
                shap_explainer = pickle.load(f)
                return shap_explainer
            
        shap_explainer = SHAPIndividual(
            self.model,
            background_data=self.combined_features[:1000],
            bow_feature_names=self.bow_feature_names,
            meta_feature_names=self.meta_feature_names,
        )
        # Save SHAP Explainer for future use
        with open(cache, "wb") as f:
            pickle.dump(shap_explainer, f)
            
        return shap_explainer
    
    def retrieve_shap_values(self, use_cache=True) -> SHAPIndividual:
        
        cache = "data/shap.csv"
        
        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached shap values...")
                explanations = json.load(f)
                return explanations
        
        explainer = self.retrieve_shap_explainer()
        explainer.explain(self.combined_features)
        
        tokens = list(map(self.extractor.vectorizer.build_tokenizer(), self.statements))
        combined_feature_names = list(self.bow_feature_names) + list(self.meta_feature_names)
        shap_valuess = explainer.shap_values
        
        explanations = []
        
        for i, shap_values in enumerate(shap_valuess):
            i_tokens = tokens[i]
            i_tokens = [token.lower() for token in i_tokens]
            pred = self.predictions[i]
            elem_id = self.ids[i]
            explanations.append({
                "id": elem_id,
                "prediction": pred,
                "values": {}
            })
            for j, elem in enumerate(shap_values):
                if combined_feature_names[j] in i_tokens or combined_feature_names[j] in self.meta_feature_names:
                    value_key = combined_feature_names[j]
                    explanations[-1]["values"][value_key] = elem[pred]
                
        with open(cache, "w") as f:
            json.dump(explanations, f, indent=4)
        
        return explanations
        
    def retrieve_partial_dependences(self, use_cache=True, grid_resolution=20):
        
        cache = "data/pdp.csv"
            
        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached pdp values...")
                explanations = json.load(f)
                return explanations
            
        partial_dependences = []
        
        n_bow_features = len(self.bow_feature_names)
        
        for i, feature_name in enumerate(self.meta_feature_names):
            results = partial_dependence(self.model, self.combined_features, [n_bow_features + i], grid_resolution=grid_resolution)
            
            # Convert ndarrays to python list
            for key, value in results.items():
                tmp_list = list(value)
                for i, elem in enumerate(tmp_list):
                    tmp_list[i] = elem.tolist()
                    # new_item_list = []
                    # for item in tmp_list[i]:
                    #     new_item_list.append(float(item))
                results[key] = tmp_list
            
            print(results)
            
            partial_dependences.append({
                "feature": feature_name,
                "partial_dependence": results
            })
                        
        with open(cache, "w") as f:
            json.dump(partial_dependences, f, indent=4)
            
        return partial_dependences
    
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