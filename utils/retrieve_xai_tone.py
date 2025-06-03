import pickle
import os
import json
import pandas as pd
import shap

from sklearn.inspection import partial_dependence

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from utils.retrieve_model import retrieve_model


data_folder = "data_tone"


class XAIRetriever:

    def __init__(self):
        self.df = pd.read_csv(f"{data_folder}/tone_rf_train.csv")
        self.model = retrieve_model("tone_model_rf.pkl")
        self.train_features = [
            "us_vs_them_lang",
            "exaggerated_uncertainty",
            "source_quality",
            "victim_villain_language",
            "black_and_white_language",
            "dehumanization",
            "emotionality",
            "reading_difficulty",
            "sentiment",
            "polarization"
        ]

    def retrieve_confusion(self, use_cache=True):

        cache = f"{data_folder}/confusion.csv"

        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached confusion matrix...")
                shap_explainer = json.load(f)
                return shap_explainer

        labels = self.df["new_label"]
        predictions = self.df["prediction"]

        confusion = confusion_matrix(
            labels, predictions, normalize="true", labels=[0, 1, 2]
        )
        confusion = confusion.tolist()

        with open(cache, "w") as f:
            json.dump(confusion, f, indent=4)

        return confusion

    def retrieve_metrics(self, use_cache=True):

        cache = f"{data_folder}/metrics.csv"

        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached metric scores...")
                shap_explainer = json.load(f)
                return shap_explainer

        labels = self.df["new_label"]
        predictions = self.df["prediction"]
        
        probas = self.model.predict_proba(self.df[self.train_features])

        scores = {
            "accuracy": balanced_accuracy_score(labels, predictions),
            "f1_score": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
            "roc_auc": roc_auc_score(
                labels, probas, average="weighted", multi_class="ovr"
            ),
        }

        with open(cache, "w") as f:
            json.dump(scores, f, indent=4)

        return scores

    def retrieve_feature_importance(self, use_cache=True):

        cache = f"{data_folder}/feature_importance.csv"
        shaps = f"{data_folder}/shap.csv"
        
        if os.path.exists(shaps):
            with open(shaps, "rb") as f:
                shaps = json.load(f)
        else:
            raise FileNotFoundError("Please generate shap values first.")

        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached feature importance...")
                feature_importances = json.load(f)
                return feature_importances

        shap_sums = {
            "class_0": {},
            "class_1": {},
            "class_2": {},
        }
        
        for elem in shaps:
            for feature_name in self.train_features:
                if feature_name in shap_sums[f"class_{elem['prediction']}"]:
                    shap_sums[f"class_{elem['prediction']}"][feature_name] += abs(elem["values"][feature_name])
                else:
                    shap_sums[f"class_{elem['prediction']}"][feature_name] = abs(elem["values"][feature_name])

        # Create a new dictionary with normalized values per class
        normalized_dict = {}
        for class_name, metrics in shap_sums.items():
            total = sum(metrics.values())
            normalized_dict[class_name] = {
                metric: round(value / total, 2)  # Normalize and round to 2 decimals
                for metric, value in metrics.items()
            }
            
        with open(cache, "w") as f:
            json.dump(normalized_dict, f, indent=4)

        return normalized_dict

    def _retrieve_shap_explainer(self, use_cache=True):

        cache = "model/shap_explainer.pkl"

        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached shap explainer...")
                shap_explainer = pickle.load(f)
                return shap_explainer
            
        explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
        # Save SHAP Explainer for future use
        with open(cache, "wb") as f:
            pickle.dump(explainer, f)

        return explainer

    def retrieve_shap_values(self, use_cache=True):

        cache = f"{data_folder}/shap.csv"

        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached shap values...")
                explanations = json.load(f)
                return explanations

        explainer = self._retrieve_shap_explainer(use_cache)
        shap_valuess = explainer.shap_values(self.df[self.train_features])

        explanations = []

        for i, shap_values in enumerate(shap_valuess):
            pred = self.df["prediction"][i]
            elem_id = self.df["id"][i]
            statement = self.df["statement"][i]
            explanations.append({"id": int(elem_id), "prediction": int(pred), "statement": statement, "values": {}})
            for j, elem in enumerate(shap_values):
                value_key = self.train_features[j]
                explanations[-1]["values"][value_key] = float("{:.4f}".format(elem[pred]))

        print(explanations)
        
        with open(cache, "w") as f:
            json.dump(explanations, f, indent=4)

        return explanations

    def retrieve_partial_dependences(self, use_cache=True, grid_resolution=20):

        cache = f"{data_folder}/pdp.csv"

        if os.path.exists(cache) and use_cache:
            with open(cache, "rb") as f:
                print("Loading cached pdp values...")
                explanations = json.load(f)
                return explanations

        partial_dependences = []

        for i, feature_name in enumerate(self.train_features):
            results = partial_dependence(
                self.model,
                self.df[self.train_features],
                [i],
                grid_resolution=grid_resolution,
            )

            # Convert ndarrays to python list
            for key, value in results.items():
                tmp_list = list(value)
                for i, elem in enumerate(tmp_list):
                    tmp_list[i] = elem.tolist()
                results[key] = tmp_list

            print(results)

            partial_dependences.append(
                {"feature": feature_name, "partial_dependence": results}
            )

        with open(cache, "w") as f:
            json.dump(partial_dependences, f, indent=4)

        return partial_dependences
