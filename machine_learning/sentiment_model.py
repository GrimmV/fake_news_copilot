from transformers import pipeline

class SentimentModel:
    
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
        
    def generate(self, data):
        predictions = self.sentiment_pipeline(data)
        pred_scores = list(map(lambda x: x["score"], predictions))
        return pred_scores
        
# Example usage
if __name__ == "__main__":

    model = SentimentModel()
    data = ["I love you", "I hate you"]
    print(model.generate(data))