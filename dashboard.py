import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from visualize import Autoencoder, plot_features
from evaluate import save_visualizations
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# Title
st.title("Interactive Anomaly Detection by Fathia Alfajr")

# Sidebar for user inputs
st.sidebar.header("Configuration")
test_data_path = st.sidebar.text_input("Test Data Path", value="/Users/fathialfajr/.cache/kagglehub/datasets/sampadab17/network-intrusion-detection/versions/1/Test_data.csv")
models_dir = st.sidebar.text_input("Models Directory", value="saved_models")
anomaly_threshold = st.sidebar.slider("Anomaly Threshold Quantile", 0.90, 0.99, 0.95, step=0.01)
evaluation_mode = st.sidebar.selectbox("Evaluation Mode", ["Unsupervised", "Supervised"])

# Function to list available models
def list_models(models_dir):
    models = [f for f in os.listdir(models_dir) if f.endswith(".pth")]
    if not models:
        st.error("No models found in the specified directory.")
        st.stop()
    return models

# Step 1: Model Selection
try:
    models = list_models(models_dir)
    selected_model = st.sidebar.selectbox("Select a Model", models)
    model_path = os.path.join(models_dir, selected_model)
except Exception as e:
    st.error(f"Error listing models: {e}")
    st.stop()

# Step 2: Load and preprocess dataset
try:
    test_df = pd.read_csv(test_data_path)
    test_df.dropna(inplace=True)

    # Preprocess test features
    test_features = test_df.copy()
    categorical_columns = test_features.select_dtypes(include=['object']).columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        test_features[col] = label_encoder.fit_transform(test_features[col])

    scaler = MinMaxScaler()
    test_features_scaled = scaler.fit_transform(test_features)

    st.write("✅ Dataset loaded and preprocessed successfully.")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Step 3: Load selected model
try:
    input_dim = test_features_scaled.shape[1]
    autoencoder = Autoencoder(input_dim)
    autoencoder.load_state_dict(torch.load(model_path, weights_only=True))
    autoencoder.eval()
    st.write(f"✅ Model loaded successfully from: `{model_path}`")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Step 4: Evaluation based on mode
if evaluation_mode == "Unsupervised":
    st.subheader("Unsupervised Evaluation")
    try:
        features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
        with torch.no_grad():
            reconstruction_error = torch.mean((features_tensor - autoencoder(features_tensor)) ** 2, dim=1).numpy()

        threshold_value = np.quantile(reconstruction_error, anomaly_threshold)
        anomalies = reconstruction_error > threshold_value

        st.write(f"Threshold Value: {threshold_value:.4f}")
        st.write(f"Total Anomalies Detected: {anomalies.sum()} out of {len(reconstruction_error)} samples")
        
        # Add anomalies to DataFrame and display
        test_df['Anomaly'] = anomalies.astype(int)
        st.write(test_df.head())

        # Export CSV option
        if st.button("Export Anomaly Results to CSV"):
            output_path = st.text_input("Enter Output Path", value="anomaly_results.csv")
            test_df.to_csv(output_path, index=False)
            st.write(f"✅ Results exported to `{output_path}`.")
        
        if st.sidebar.button("Save Unsupervised Visualizations"):
            save_visualizations(reconstruction_error)
            st.write("✅ Visualizations saved successfully.")
    except Exception as e:
        st.error(f"Error during unsupervised evaluation: {e}")

elif evaluation_mode == "Supervised":
    st.subheader("Supervised Evaluation")
    try:
        # Check for labels
        if 'class' not in test_df.columns:
            st.error("Dataset must contain a 'class' column for supervised evaluation.")
            st.stop()

        labels = test_df['class'].apply(lambda x: 1 if x == "anomaly" else 0).values
        features_tensor = torch.tensor(test_features_scaled, dtype=torch.float32)
        with torch.no_grad():
            reconstruction_error = torch.mean((features_tensor - autoencoder(features_tensor)) ** 2, dim=1).numpy()

        threshold_value = np.quantile(reconstruction_error, anomaly_threshold)
        predictions = (reconstruction_error > threshold_value).astype(int)

        # Classification report
        st.write("Classification Report:")
        st.text(classification_report(labels, predictions))

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        # Display confusion matrix metrics
        st.write("Confusion Matrix Metrics:")
        st.write(f"True Positives (TP): {tp}")
        st.write(f"True Negatives (TN): {tn}")
        st.write(f"False Positives (FP): {fp}")
        st.write(f"False Negatives (FN): {fn}")

        # Add bar chart visualization for metrics
        metrics_data = {
            "Metric": ["True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)"],
            "Count": [tp, tn, fp, fn]
        }
        metrics_df = pd.DataFrame(metrics_data)
        st.write("Confusion Matrix Metrics (Bar Chart):")
        st.bar_chart(metrics_df.set_index("Metric"))

        # Add predictions to DataFrame
        test_df['Predictions'] = predictions
        st.write(test_df.head())

        # Export CSV option
        if st.button("Export Evaluation Results to CSV"):
            output_path = st.text_input("Enter Output Path", value="evaluation_results.csv")
            test_df.to_csv(output_path, index=False)
            st.write(f"✅ Results exported to `{output_path}`.")

        # Confusion Matrix
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(labels, predictions, ax=ax, cmap="Blues")
        st.pyplot(fig)

        if st.sidebar.button("Save Supervised Visualizations"):
            save_visualizations(reconstruction_error, labels)
            st.write("✅ Visualizations saved successfully.")
    except Exception as e:
        st.error(f"Error during supervised evaluation: {e}")

# Step 5: Visualize results
st.subheader("Visualizations")

# Column selection for scatter plot
available_columns = test_df.columns.tolist()
col_x = st.selectbox("Select X-axis column:", available_columns, index=0)
col_y = st.selectbox("Select Y-axis column:", available_columns, index=1)

if st.button("Generate Scatter Plot"):
    try:
        fig = plt.figure()
        plot_features(test_features, test_features_scaled, test_df.get('Anomaly', np.zeros(len(test_features))), col_x, col_y)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating plot: {e}")
