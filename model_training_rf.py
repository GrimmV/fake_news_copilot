import datasets
import pandas as pd
from machine_learning.rf.rf_model import RandomForestTextClassifier
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

    # Initialize and train the classifier
    classifier = RandomForestTextClassifier(n_estimators=100, random_state=42)
    classifier.train(combined_features, y_train)

    # Evaluate the classifier
    accuracy_train = classifier.evaluate(combined_features, y_train)
    print(f"Accuracy in training: {accuracy_train:.2f}")
