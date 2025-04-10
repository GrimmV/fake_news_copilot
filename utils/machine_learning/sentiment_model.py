from transformers import pipeline

class SentimentModel:
    
    def __init__(self):
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", top_k=None)
        
    def generate(self, data):
        predictions = self.sentiment_pipeline(data)
        print(predictions)
        pred_scores = list(map(lambda x: self._convert_to_continuous(x), predictions))
        return pred_scores
        
    def _convert_to_continuous(self, all_scores):
        score = 0
        for elem in all_scores:
            if (elem["label"] == "LABEL_2"):
                score += elem["score"]
            elif (elem["label"] == "LABEL_0"):
                score -= elem["score"]
            elif (elem["label"] == "LABEL_1"):
                score *= 1.0-elem["score"]
        return score
        
# Example usage
if __name__ == "__main__":

    model = SentimentModel()
    data = ["I love you", "I hate you"]
    print(model.generate(data))