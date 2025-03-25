import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from machine_learning.rf.text_feature_extractor import TextFeatureExtractor


def retrieve_data():
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])

    X_train, X_test, y_train, y_test = train_test_split(
        train_raw["statement"].to_list(),
        train_raw["label"].to_list(),
        test_size=0.2,
        random_state=42,
    )

    extractor = TextFeatureExtractor(X_train)

    bow_features = extractor.extract_bow_features(X_train)
    meta_features, meta_feature_names = extractor.extract_meta_features(X_train)

    bow_feature_names = extractor.vectorizer.get_feature_names_out()

    # Combine features
    combined_features = np.hstack((bow_features, meta_features))

    return (
        bow_features,
        bow_feature_names,
        meta_features,
        meta_feature_names,
        combined_features,
    )
