import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import os
import glob

import time

from sklearn.metrics import classification_report, roc_auc_score

# Path to the dataset files || changeable
test_data_path = '/Users/fathialfajr/.cache/kagglehub/datasets/sampadab17/network-intrusion-detection/versions/1/Test_data.csv'

# Step 2: Load and preprocess dataset
test_df = pd.read_csv(test_data_path)
test_df.dropna(inplace=True)

# Preprocess test features
test_features = test_df
categorical_columns = test_features.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_columns:
    test_features[col] = label_encoder.fit_transform(test_features[col])

scaler = MinMaxScaler()
test_features_scaled = scaler.fit_transform(test_features)

# Load trained model

def get_latest_model_path(models_dir):
    model_files = glob.glob(os.path.join(models_dir, "autoencoder_epoch*.pth"))
    if not model_files:
        raise FileNotFoundError("No model files found in the specified directory.")
    latest_model = max(model_files, key=os.path.getctime)
    return latest_model

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

input_dim = test_features_scaled.shape[1]
autoencoder = Autoencoder(input_dim)
models_dir = "saved_models"  # Change to your model directory path

try:
    model_path = get_latest_model_path(models_dir)
    print(f"Loading model: {model_path}")
    autoencoder.load_state_dict(torch.load(model_path, weights_only=True))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

autoencoder.eval()

# Step 4: Evaluate on test set
test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
with torch.no_grad():
    reconstructed = autoencoder(test_tensor)
    reconstruction_error = torch.mean((test_tensor - reconstructed) ** 2, dim=1)

# Define anomaly threshold and predict anomalies
threshold = torch.quantile(reconstruction_error, 0.95)
test_predictions = (reconstruction_error > threshold).int()

# Add predictions to the test dataframe
test_df['Anomaly'] = test_predictions.numpy()
print(test_df.head())

# Step 5: Visualize results
def plot_features(features, features_scaled, predictions, col_x, col_y):
    if col_x not in features.columns or col_y not in features.columns:
        raise ValueError(f"One or both columns '{col_x}' and '{col_y}' not found in features DataFrame.")

    x_index = features.columns.get_loc(col_x)
    y_index = features.columns.get_loc(col_y)

    plt.scatter(features_scaled[:, x_index], features_scaled[:, y_index], c=predictions, cmap='coolwarm')
    plt.title(f'Anomaly Detection ({col_x} vs {col_y})')
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.show()

# User input for visualization
print("Available features for visualization:")
print(test_features.columns.tolist())

# Prompt the user to select columns with validation loop
def get_valid_column(features, prompt):
    while True:
        col = input(prompt)
        if col in features.columns:
            return col
        else:
            print("Invalid input! Please enter a valid column name from the list.")

col_x = get_valid_column(test_features, "Enter the column name for the x-axis: ")
col_y = get_valid_column(test_features, "Enter the column name for the y-axis: ")

try:
    print("Visualizing results...")
    plot_features(test_features, test_features_scaled, test_df['Anomaly'], col_x, col_y)

except ValueError as e:
    print("Error:", e)


