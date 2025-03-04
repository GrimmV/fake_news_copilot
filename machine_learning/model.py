import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FakeNewsClassifier(nn.Module):
    def __init__(self, num_tabular_features, bert_model_name="bert-base-uncased"):
        super(FakeNewsClassifier, self).__init__()
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # Typically 768 for BERT-base
        
        # Tabular feature processing
        self.tabular_fc = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer (BERT + Tabular features)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 5),  # Binary classification (fake or real)
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask, tabular_features):
        # BERT processing
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_output.pooler_output  # [batch_size, 768]
        
        # Tabular feature processing
        tabular_embedding = self.tabular_fc(tabular_features)  # [batch_size, 64]
        
        # Concatenation
        combined_features = torch.cat((text_embedding, tabular_embedding), dim=1)  # [batch_size, 768+64]
        
        # Classification
        output = self.classifier(combined_features)
        return output
    
    # Preprocessing: Scaling numerical values & one-hot encoding categorical values
    def preprocessing(self, df, numerical_features: list, categorical_features: list):
        
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical_features),  
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  
        ])

        # Apply transformations
        processed_features = preprocessor.fit_transform(df)
        
        tabular_tensor = torch.tensor(processed_features.toarray(), dtype=torch.float32)
        return tabular_tensor
    
# Example usage
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Simulated inputs
    text_samples = ["Fake news example", "Real news article"]
    encoded_inputs = tokenizer(text_samples, padding=True, truncation=True, max_length=512, return_tensors="pt")
    
    tabular_features = torch.rand(len(text_samples), 10)  # Assume 10 tabular features
    
    # Model initialization
    model = FakeNewsClassifier(num_tabular_features=10)
    
    # Forward pass
    with torch.no_grad():
        output = model(encoded_inputs["input_ids"], encoded_inputs["attention_mask"], tabular_features)
    
    print(output)  # Probabilities for fake news detection