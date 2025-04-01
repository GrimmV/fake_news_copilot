from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from typing import List, Dict


class SimilarityHandler:

    def __init__(self, phrases: List[Dict]):
        # Load model for similarity analysis
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.phrase_list = list(map(lambda x: x["statement"], phrases))
        self.phrase_ids = list(map(lambda x: x["id"], phrases))
        self.phrase_preds = list(map(lambda x: x["predictions"], phrases))

        self.phrase_embeddings = self.model.encode(self.phrase_list)

    def find_most_similar_phrases(self, query, top_n=5):
        """Finds the most similar phrases to the given query."""
        query_embedding = self.model.encode([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.phrase_embeddings)[0]

        # Get top N most similar phrases efficiently
        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]

        return [
            {
                "id": self.phrase_ids[i],
                "phrase": self.phrase_list[i],
                "predictions": self.phrase_preds[i],
                "score": float(similarities[i]),
            }
            for i in top_indices
        ]

    def find_nth_most_similar_phrase(self, query, n):
        """Finds the nth most similar phrase to the given query.

        Args:
            query: The input string to compare against
            n: The rank of similarity to return (0 for most similar, 1 for second most, etc.)

        Returns:
            A dictionary containing the id, phrase, and similarity score of the nth most similar phrase
            Returns None if n is out of bounds
        """

        if n < 0:
            return None

        query_embedding = self.model.encode([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.phrase_embeddings)[0]

        # Get all indices sorted by similarity in descending order
        sorted_indices = np.argsort(similarities)[::-1]

        # Check if n is within valid range
        if n > len(sorted_indices) - 1:
            return None

        # Get the nth most similar phrase (n-1 because of zero-based indexing)
        idx = sorted_indices[n]

        return {
            "id": self.phrase_ids[idx],
            "phrase": self.phrase_list[idx],
            "predictions": self.phrase_preds[idx],
            "score": float(similarities[idx]),
        }
