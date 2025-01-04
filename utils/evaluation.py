import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_reconstruction_error(reconstruction_error, threshold):
    """Plot reconstruction error distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_error, bins=50, alpha=0.7, label="Reconstruction Error")
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

def plot_confusion_matrix(ground_truth, predictions):
    """Plot confusion matrix."""
    cm = confusion_matrix(ground_truth, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
