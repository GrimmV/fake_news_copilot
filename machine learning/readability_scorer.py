import textstat
import spacy
import nltk
from collections import Counter
from nltk.corpus import words

# Download English words dataset
nltk.download('words')
english_words = set(words.words())

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class ReadabilityScorer:
    
    def __init__(self):
        pass
    
    def analyze_text_complexity(self, text):
        doc = nlp(text)
        
        # Token and word-level stats
        words = [token.text for token in doc if token.is_alpha]
        unique_words = set(words)
        num_words = len(words)
        
        # Lexical Diversity (TTR)
        ttr = len(unique_words) / num_words if num_words > 0 else 0
        
        # Average Word Length
        avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
        
        # Syllable Count per Word
        syllables = [textstat.syllable_count(word) for word in words]
        avg_syllables_per_word = sum(syllables) / num_words if num_words > 0 else 0
        
        # Approximate Dale-Chall Score (counts "hard words" not in common vocabulary)
        difficult_words = [word for word in words if word.lower() not in english_words]
        difficult_word_ratio = len(difficult_words) / num_words if num_words > 0 else 0

        # Syntactic Complexity: Dependency Parsing Depth
        dependency_depth = max([len(list(token.ancestors)) for token in doc], default=0)
        
        return {
            "Lexical Diversity (TTR)": round(ttr, 3),
            "Average Word Length": round(avg_word_length, 2),
            "Avg Syllables per Word": round(avg_syllables_per_word, 2),
            "Difficult Word Ratio": round(difficult_word_ratio, 2),
            "Dependency Depth": dependency_depth,
        }
    
    
# Example usage
if __name__ == "__main__":

    model = ReadabilityScorer()
    text = "I love you"
    print(model.analyze_text_complexity(text))