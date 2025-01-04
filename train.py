import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time

# File paths
train_data_path = '/path/to/train_data.csv'
output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_and_preprocess_data(file_path, supervised=True):
    """Load and preprocess data."""
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)

    if supervised and 'class' not in data.columns:
        raise ValueError("'class' column is required for supervised training.")

    labels = data['class'].map({'normal': 1, 'anomaly': 0}).values if 'class' in data.columns else None
    features = data.drop(columns=['class'], errors='ignore')

    # Encode categorical columns
    categorical_columns = features.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        features[col] = label_encoder.fit_transform(features[col])

    # Scale features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, labels

def save_model(autoencoder, epoch, mode):
    """Save model with user choice."""
    model_name = f"{mode}_autoencoder_epoch{epoch}_{time.strftime('%Y%m%d-%H%M%S')}.pth"
    model_path = os.path.join(output_dir, model_name)

    existing_models = [f for f in os.listdir(output_dir) if f.startswith(f"{mode}_autoencoder")]
    if existing_models:
        print("Existing models:")
        for i, model in enumerate(existing_models, 1):
            print(f"{i}. {model}")
        print("Choose an option:")
        print("1. Overwrite the latest model.")
        print("2. Save a new model with a timestamp.")
        choice = input("Enter your choice (1/2): ").strip()

        if choice == "1":
            model_path = os.path.join(output_dir, existing_models[-1])
            print(f"Overwriting model: {model_path}")
        elif choice != "2":
            print("Invalid choice. Saving with a new timestamp.")

    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

def train(features, labels, supervised=True):
    """Train the model."""
    input_dim = features.shape[1]
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32)) if not supervised else \
              TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    num_epochs = 50
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in loader:
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(loader):.4f}")

    save_model(autoencoder, num_epochs, "supervised" if supervised else "unsupervised")

if __name__ == "__main__":
    import argparse

    # Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["supervised", "unsupervised"], required=True, help="Training mode")
    args = parser.parse_args()

    supervised_mode = args.mode == "supervised"
    features, labels = load_and_preprocess_data(train_data_path, supervised=supervised_mode)

    train(features, labels, supervised=supervised_mode)
