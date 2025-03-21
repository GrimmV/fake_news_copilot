import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from machine_learning.rf.rf_model import RandomForestTextClassifier

if __name__ == "__main__":
    # Load training dataset
    dataset = "chengxuphd/liar2"
    dataset = datasets.load_dataset(dataset)
    train_raw = pd.DataFrame(dataset["train"])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(train_raw['statement'].to_list(), train_raw['label'].to_list(), test_size=0.0, random_state=42)
    
    # Initialize and train the classifier
    classifier = RandomForestTextClassifier(n_estimators=100, random_state=42, train_text=X_train)
    classifier.train(X_train, y_train)
    
    # Evaluate the classifier
    accuracy_train = classifier.evaluate(X_train, y_train, name = "eval_train")
    # accuracy_test = classifier.evaluate(X_test, y_test, name = "eval_test")
    print(f'Accuracy in training: {accuracy_train:.2f}')
    # print(f'Accuracy in test data: {accuracy_test:.2f}')