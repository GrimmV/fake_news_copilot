from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle


class RandomForestToneClassifier:
    """Class to train and evaluate a RandomForest classifier for text classification."""
    
    def __init__(self, n_estimators=100, random_state=42, max_depth=12):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    
    def train(self, X, y, cache_file: str = "model/tone_model_rf.pkl"):
        
        """Train the RandomForest classifier."""
        self.model.fit(X, y)
        # Save processed Model for future use
        with open(cache_file, "wb") as f:
            pickle.dump(self.model, f)
    
    def evaluate(self, X, y):
        """Evaluate the classifier on test data."""
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        return accuracy, f1