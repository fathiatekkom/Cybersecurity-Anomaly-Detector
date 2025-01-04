import kagglehub
import os

# Path to the dataset files
train_data_path = '/Users/fathialfajr/.cache/kagglehub/datasets/sampadab17/network-intrusion-detection/versions/1/Train_data.csv'
test_data_path = '/Users/fathialfajr/.cache/kagglehub/datasets/sampadab17/network-intrusion-detection/versions/1/Test_data.csv'

# Step 1: Download the dataset
def download_dataset():
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print("Downloading dataset...")
        kagglehub.dataset_download("sampadab17/network-intrusion-detection", target_dir="./datasets")
        print("Dataset downloaded!")
    else:
        print("Dataset already exists.")

if __name__ == "__main__":
    download_dataset()
