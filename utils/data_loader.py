import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

def load_and_preprocess_data(file_path, label_column="class"):
    """Load and preprocess data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load dataset
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)

    # Encode categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Scale features
    scaler = MinMaxScaler()
    features = data.drop(columns=[label_column], errors="ignore")
    features_scaled = scaler.fit_transform(features)

    # Extract labels if available
    labels = data[label_column].map({"normal": 1, "anomaly": 0}).values if label_column in data.columns else None

    return features_scaled, labels
