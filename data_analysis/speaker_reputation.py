import pandas as pd
from config import base_columns

class SpeakerReputationHandler:
    
    def __init__(self):
        pass
    
    def reputation(self, speaker, df):
        
            
        # Filter dataframe by the given speaker
        speaker_df = df[df['speaker'] == speaker]
        
        # Count occurrences of each label
        label_counts = speaker_df['label'].value_counts(normalize=True) * 100
        
        # Convert to dictionary and fill missing labels with 0%
        all_labels = ['pants-fire', 'barely-true', 'half-true', 'mostly-true', 'false', 'true']
        label_percentages = {label: round(float(label_counts.get(label, 0)), 2) for label in all_labels}
        
        return label_percentages
    
    
# Example usage
if __name__ == "__main__":

    handler = SpeakerReputationHandler()
    df = pd.read_csv("data/train.tsv", sep='\t')
    df.columns = base_columns
    print()
    print("Barack Obama:\n")
    print(handler.reputation("barack-obama", df))
    print()
    print("Donald Trump:\n")
    print(handler.reputation("donald-trump", df))