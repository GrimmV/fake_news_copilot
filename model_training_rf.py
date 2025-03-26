import datasets
import pandas as pd
from machine_learning.rf.rf_model import RandomForestTextClassifier
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
    ) = data_retriever.generate_input_data(X_train, y_train, name="train")

    # Initialize and train the classifier
    classifier = RandomForestTextClassifier(n_estimators=100, random_state=42)
    classifier.train(combined_features, y_train)

    # Evaluate the classifier
    accuracy_train = classifier.evaluate(combined_features, y_train)
    print(f"Accuracy in training: {accuracy_train:.2f}")
