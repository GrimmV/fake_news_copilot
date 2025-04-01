
from utils.xai.proximity_based.similarity_handler import SimilarityHandler

class CounterfactualHandler:
    
    def __init__(self, similarity_handler: SimilarityHandler):
        
        self.similarity_handler = similarity_handler
        
    def find_counterfactuals(self, query, pred, k = 3):
        
        counterfactuals = []
        
        # initially take the 10 most similar phrases
        n = 10
        first_n_similar = self.similarity_handler.find_most_similar_phrases(query, n)
        
        for elem in first_n_similar:
            if elem["predictions"] != pred:
                counterfactuals.append(elem)
        
        # if number of counterfactuals is smaller than 3, keep going.
        while len(counterfactuals) < k:
            self.similarity_handler.find_nth_most_similar_phrase(n)
            n += 1
            
        return counterfactuals