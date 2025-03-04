import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

class FakeNewsClassifier(nn.Module):
    def __init__(self, num_tabular_features, numerical_features, categorical_features, bert_model_name="bert-base-uncased"):
        super(FakeNewsClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # Typically 768 for BERT-base
        self.optimizer = optim.Adam(self.parameters(), lr=2e-5, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for 5 classes
        num_classes = 6
        
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
            nn.Linear(128, num_classes),  # Binary classification (fake or real)
        )
        
        # Store feature names
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        # Define preprocessor pipeline
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])
        
        self.to(self.device)
        
    def preprocess_tabular(self, df):
        """Preprocess raw tabular data into a tensor."""
        processed_features = self.preprocessor.fit_transform(df)  # Ensure fitted before inference
        return torch.tensor(processed_features, dtype=torch.float32).to(self.device)

    def forward(self, input_text, tabular_df):
        """Full forward pass, including text tokenization and tabular preprocessing."""
        
        # Preprocess tabular data
        tabular_tensor = self.preprocess_tabular(tabular_df)

        # Tokenize text and get BERT embeddings
        encoded_input = self.tokenizer(input_text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        bert_output = self.bert(**encoded_input).last_hidden_state[:, 0, :]  # Extract CLS token
        
        # Process tabular data
        tabular_features = self.tabular_fc(tabular_tensor)

        # Concatenate BERT and tabular features
        combined_features = torch.cat((bert_output, tabular_features), dim=1)
        
        # Classification
        return self.classifier(combined_features)
    
    def train_model(self, dataloader, num_epochs=3):
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in dataloader:
                input_text = batch["input_text"]
                tabular_df = batch["tabular_df"]
                labels = batch["label"].to(self.device).long()  # Reshape for BCEWithLogitsLoss

                self.optimizer.zero_grad()
                outputs = self.forward(input_text, tabular_df)
                
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            epoch_info = f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
            logger.info(epoch_info)
            print(epoch_info)
    
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
