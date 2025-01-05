# Anomaly Detection for Cybersecurity Data Streams

**Version 1.0**

---

## Overview

This project leverages **Machine Learning** to detect anomalies in cybersecurity data streams, providing a robust and scalable tool for identifying potential threats. By combining an interactive user interface and advanced metrics, this project simplifies anomaly detection for cybersecurity practitioners and students.

---

## Features

- **Unsupervised Learning with Autoencoders**: Detect anomalies using reconstruction errors in unlabeled datasets.
- **Interactive Dashboard**:
  - Upload datasets for analysis.
  - Select models and adjust thresholds for anomaly detection.
  - Visualize anomalies with scatter plots and customizable axes.
- **Metrics and Transparency**:
  - Provides ROC curves, precision, false positives/negatives, and other evaluation metrics.
  - Save models with timestamps for reproducibility.
- **Extensible Design**:
  - Modular codebase for easy updates and further development.

---

## Project Structure

- **`train.py`**: Script for training the autoencoder on labeled or unlabeled datasets.
- **`inference.py`**: Performs anomaly detection using trained models.
- **`evaluate.py`**: Calculates performance metrics such as precision, recall, and ROC.
- **`visualize.py`**: Generates interactive visualizations for anomaly detection results.
- **`dashboard.py`**: Streamlit-based dashboard for user interaction.

---

## How to Run

### Prerequisites

- **Python Version**: 3.8 or later.
- **Libraries**:
  - `pandas`
  - `numpy`
  - `torch`
  - `scikit-learn`
  - `streamlit`

### Steps

1. Clone the repository:
   ```bash
   git clone git@github.com:your-username/your-repo.git
2. Navigate to the project directory:
   ```bash
   cd your-repo
3. Install dependencies
   ```bash
   pip install -r requirements.txt
4. Train a model:
   ```bash
   python train.py
5. Run the dashboard
   ```bash
   streamlit run dashboard.py

### Usage
- Upload your dataset through the dashboard.
- Select a saved model for inference.
- Adjust anomaly detection thresholds and visualize results interactively.
- View evaluation metrics and export anomaly results as a CSV file.

## Documentation

For detailed technical information, refer to the [Project Documentation](./Project_Documentation_Cybersecurity_Anomaly_Detection.pdf).

---

## Acknowledgments

Special thanks to:

- The developers of **PyTorch**, **Pandas**, and **Streamlit** for providing excellent libraries.
- Inspiration from research papers on anomaly detection in cybersecurity.


### **Key Notes**
1. Save this as `README.md` in the root of your project directory.
2. Update the placeholders like `git@github.com:your-username/your-repo.git` and `LICENSE` file link based on your project repository.
3. If you don’t have a `requirements.txt`, generate one:
   ```bash
   pip freeze > requirements.txt


