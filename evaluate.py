import os
import time
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay,
    roc_curve, RocCurveDisplay
)
from utils.data_loader import load_and_preprocess_data
from utils.model import Autoencoder
from utils.evaluation import plot_reconstruction_error
from utils.config import test_data_path, models_dir, anomaly_threshold

output_dir = "saved_models"
os.makedirs(output_dir, exist_ok=True)

def select_model():
    """List available models and allow user to select one."""
    models = [f for f in os.listdir(output_dir) if f.endswith(".pth")]
    if not models:
        raise FileNotFoundError("No models found in the saved_models directory.")

    print("Available models:")
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    choice = input("Enter the number of the model you want to evaluate: ").strip()

    try:
        model_path = os.path.join(output_dir, models[int(choice) - 1])
        print(f"Selected model: {model_path}")
        return model_path
    except (IndexError, ValueError):
        raise ValueError("Invalid selection. Please enter a valid number.")

def save_classification_report(ground_truth, predictions, output_dir="results", filename="classification_report"):
    """Save classification report to a file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}.txt")

    if os.path.exists(output_path):
        print(f"The file '{output_path}' already exists. Choose an option:")
        print("1. Keep the old file (do not overwrite).")
        print("2. Overwrite the file.")
        print("3. Save as a new file with a timestamp.")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == "1":
            print("Keeping the old file. No new file saved.")
            return
        elif choice == "2":
            print(f"Overwriting the file: {output_path}")
        elif choice == "3":
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = os.path.join(output_dir, f"{filename}_{timestamp}.txt")
            print(f"Saving as a new file: {output_path}")
        else:
            print("Invalid choice. Keeping the old file.")
            return

    report = classification_report(ground_truth, predictions)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to: {output_path}")


def save_visualizations(reconstruction_error, ground_truth=None, output_dir="results"):
    """Save visualizations like reconstruction error, precision-recall curve, and ROC curve."""
    os.makedirs(output_dir, exist_ok=True)

    # Save Reconstruction Error Distribution
    output_path = os.path.join(output_dir, "reconstruction_error_distribution.png")
    plot_reconstruction_error(reconstruction_error, anomaly_threshold)
    plt.savefig(output_path)
    print(f"Reconstruction error distribution saved to: {output_path}")
    plt.close()

    if ground_truth is not None:
        # Precision-Recall Curve
        disp = PrecisionRecallDisplay.from_predictions(ground_truth, reconstruction_error)
        disp.plot()
        output_path = os.path.join(output_dir, "precision_recall_curve.png")
        plt.title("Precision-Recall Curve")
        plt.savefig(output_path)
        print(f"Precision-recall curve saved to: {output_path}")
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(ground_truth, reconstruction_error)
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
        output_path = os.path.join(output_dir, "roc_curve.png")
        plt.title("ROC Curve")
        plt.savefig(output_path)
        print(f"ROC curve saved to: {output_path}")
        plt.close()


def evaluate_supervised(features, labels, model):
    """Supervised evaluation with ground truth."""
    with torch.no_grad():
        reconstruction_error = ((torch.tensor(features) - model(torch.tensor(features))) ** 2).mean(dim=1).numpy()

    predictions = (reconstruction_error > anomaly_threshold).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    # Print and save classification report
    print("Classification Report:")
    print(classification_report(labels, predictions))
    save_option = input("Do you want to save the evaluation outputs? (y/n): ").strip().lower()
    if save_option == "y":
        save_classification_report(labels, predictions)
        save_visualizations(reconstruction_error, labels)

    # Plot visualizations
    ConfusionMatrixDisplay.from_predictions(labels, predictions).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    PrecisionRecallDisplay.from_predictions(labels, reconstruction_error).plot()
    plt.title("Precision-Recall Curve")
    plt.show()

    fpr, tpr, _ = roc_curve(labels, reconstruction_error)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("ROC Curve")
    plt.show()


def evaluate_unsupervised(features, model):
    """Unsupervised evaluation without ground truth."""
    features_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        reconstruction_error = (
        (torch.tensor(features, dtype=torch.float32) - model(torch.tensor(features, dtype=torch.float32))) ** 2
    ).mean(dim=1).numpy()

    # Detect anomalies based on threshold
    anomalies = reconstruction_error > anomaly_threshold
    print(f"Total anomalies detected: {anomalies.sum()} out of {len(reconstruction_error)} samples")

    # Save reconstruction error distribution
    save_option = input("Do you want to save the reconstruction error distribution? (y/n): ").strip().lower()
    if save_option == "y":
        save_visualizations(reconstruction_error)

    # Plot reconstruction error distribution
    plot_reconstruction_error(reconstruction_error, anomaly_threshold)
    plt.show()


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["supervised", "unsupervised"], required=True, help="Evaluation mode")
    args = parser.parse_args()

    
    # Load and preprocess data
    features, labels = load_and_preprocess_data(test_data_path)

    # Load trained model
    input_dim = features.shape[1]
    model = Autoencoder(input_dim)
    model_path = f"{models_dir}/{args.mode}_autoencoder.pth"

    # Dynamically select a model
    model_path = select_model()

    # Load the model
    input_dim = features.shape[1]
    model = Autoencoder(input_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))  # Add weights_only=True for the warning
    model.eval()


    # Evaluate based on mode
    if args.mode == "supervised":
        if labels is None:
            raise ValueError("Labels are required for supervised evaluation."
            "Please ensure the dataset contains a 'class' column with values like 'normal' and 'anomaly'." )
        evaluate_supervised(features, labels, model)
    elif args.mode == "unsupervised":
        evaluate_unsupervised(features, model)
