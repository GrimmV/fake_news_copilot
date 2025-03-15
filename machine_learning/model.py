import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging

logger = logging.getLogger(__name__)

class FakeNewsClassifier(nn.Module):
    def __init__(self, num_numeric_features, bert_model_name="bert-base-uncased"):
        super(FakeNewsClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # Typically 768 for BERT-base
        
        self.optimizer = optim.Adam(self.parameters(), lr=2e-5, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for 5 classes
        num_classes = 6
        
        self.batch_norm = nn.BatchNorm1d(num_numeric_features)  # Acts like StandardScaler
        
        # Tabular processing
        self.tabular_fc = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Fusion layer
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        self.to(self.device)

    def forward(self, input_text, numerical_features):
        """Differentiable forward pass"""
        
        encoded_input = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        
        print("Encoded input shape:", {k: v.shape for k, v in encoded_input.items()})
        print("Numerical features shape:", numerical_features.shape)
        
        bert_output = self.bert(**encoded_input).last_hidden_state[:, 0, :]
        
        print("Numerical features shape:", numerical_features.shape)
        print("Is contiguous:", numerical_features.is_contiguous())
        print("Dtype:", numerical_features.dtype)

        # Normalize numerical features
        x = self.batch_norm(numerical_features)
        
        print("Output of batch norm shape:", x.shape)

        tabular_features = self.tabular_fc(x)

        # Combine with BERT output
        combined_features = torch.cat((bert_output, tabular_features), dim=1)

        return self.classifier(combined_features)
    
    def train_model(self, dataloader, num_epochs=3):
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in dataloader:
                text = batch["text"]
                tabular = batch["tabular"].to(self.device)
                labels = batch["label"].to(self.device).long()  # Reshape for BCEWithLogitsLoss

                self.optimizer.zero_grad()
                outputs = self.forward(text, tabular)
                
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
