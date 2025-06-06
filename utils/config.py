base_columns = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_onfire_counts", "context"]

numerical_cols = ["Lexical Diversity (TTR)", "Average Word Length", "Avg Syllables per Word", 
                      "Difficult Word Ratio", "Dependency Depth", "Length", "sentiment"]

# Caching
use_cached_data = True
use_cached_model = True
resume_training = False

# model hyperparameters
lr = 2e-5
weight_decay = 1e-3
dropout = 0.3
bert_layers_grad = 0
num_epochs = 1

# storage

model_location = "model/model.pkl"
