import pandas as pd
from utils.machine_learning.rf.rf_tone_classifier import RandomForestToneClassifier


# Define mapping to simplify labels
def _relabel(x):
    if x in [0, 1]:
        return 0
    elif x in [2, 3]:
        return 1
    elif x in [4, 5]:
        return 2


if __name__ == "__main__":

    data = pd.read_csv("tone_prep/data_with_tone_analysis_mini.csv")

    train_cols = [
        "us_vs_them_lang",
        "exaggerated_uncertainty",
        "source_quality",
        "victim_villain_language",
        "black_and_white_language",
        "dehumanization",
        "emotionality",
        "reading_difficulty",
        "sentiment",
        "polarization",
    ]

    X_train = data[train_cols]
    y_train = data["label"]
    y_new = [_relabel(elem) for elem in y_train]

    # Initialize and train the classifier
    classifier = RandomForestToneClassifier(n_estimators=100, random_state=43)
    classifier.train(X_train, y_new)

    model = classifier.model

    # Train the model
    predictions = model.predict(X_train)
    probas = model.predict_proba(X_train)
    proba_df = pd.DataFrame(
        probas, columns=[f"prob_class_{i}" for i in range(probas.shape[1])]
    )

    data["prediction"] = predictions
    data["new_label"] = y_new

    data = pd.concat([data, proba_df], axis=1)

    # Evaluate the classifier
    accuracy_train, f1_train = classifier.evaluate(X_train, y_new)
    print(f"Accuracy in training: {accuracy_train:.2f}")
    print(f"F1 in training: {f1_train:.2f}")

    data.to_csv("data_tone/tone_rf_train.csv", index=False)
