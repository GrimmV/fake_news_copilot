import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import logging
from config import lr, weight_decay, dropout, bert_layers_grad, num_epochs

logger = logging.getLogger(__name__)

class FakeNewsClassifier(nn.Module):
    def __init__(self, num_numeric_features, bert_model_name="bert-base-uncased"):
        super(FakeNewsClassifier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_hidden_size = self.bert.config.hidden_size  # Typically 768 for BERT-base
        
        # Only use last couple of layers for BERT fine-tuning
        for param in self.bert.parameters():
            param.requires_grad = False

        # Then unfreeze only the last few layers for fine-tuning
        for param in self.bert.encoder.layer[-bert_layers_grad:].parameters():
            param.requires_grad = True
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for 5 classes
        num_classes = 6
        
        self.batch_norm = nn.BatchNorm1d(num_numeric_features)  # Acts like StandardScaler
        
        # Tabular processing
        self.tabular_fc = nn.Sequential(
            nn.Linear(num_numeric_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion layer
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self.to(self.device)

    def forward(self, x):
        """Differentiable forward pass using pre-tokenized inputs"""
        
        input_ids, attention_mask, numerical_features = x
        
        # Prepare BERT input
        bert_inputs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device)
        }

        # Get BERT output (use [CLS] token)
        bert_output = self.bert(**bert_inputs).last_hidden_state[:, 0, :]

        # Normalize and project numerical features
        x = self.batch_norm(numerical_features.to(self.device))
        tabular_features = self.tabular_fc(x)

        # Concatenate BERT and tabular features
        combined_features = torch.cat((bert_output, tabular_features), dim=1)

        # Final classification layer
        return self.classifier(combined_features)
    
    def train_model(self, train_dataloader, val_dataloader=None):
        self.train()
        for epoch in range(num_epochs):
            # --- Training ---
            self.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch in train_dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                tabular = batch["tabular"].to(self.device)
                labels = batch["label"].to(self.device).long()

                self.optimizer.zero_grad()
                outputs = self.forward((input_ids, attention_mask, tabular))

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            avg_train_loss = total_loss / len(train_dataloader)
            train_accuracy = correct / total

            # --- Validation ---
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            if val_dataloader:
                self.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        input_ids = batch["input_ids"]
                        attention_mask = batch["attention_mask"]
                        tabular = batch["tabular"].to(self.device)
                        labels = batch["label"].to(self.device).long()

                        outputs = self.forward((input_ids, attention_mask, tabular))
                        loss = self.criterion(outputs, labels)

                        val_loss += loss.item()
                        predicted = torch.argmax(outputs, dim=1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)

                avg_val_loss = val_loss / len(val_dataloader)
                val_accuracy = val_correct / val_total
            else:
                avg_val_loss = val_accuracy = None

            # --- Logging ---
            epoch_info = f"Epoch [{epoch+1}/{num_epochs}] | " \
                        f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
            if val_dataloader:
                epoch_info += f" | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}"

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
        output = model((encoded_inputs["input_ids"], encoded_inputs["attention_mask"], tabular_features))
    
    print(output)  # Probabilities for fake news detection
