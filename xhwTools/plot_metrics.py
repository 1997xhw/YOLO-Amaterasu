import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_precision(dataframes, labels):
    """
    Plot precision vs. epoch for multiple DataFrames on the same plot.
    Args:
    - dataframes: List of DataFrames, each containing 'epoch' and 'metrics/precision(B)' columns.
    - labels: List of labels for each DataFrame, to use in the plot legend.
    """
    plt.figure(figsize=(8, 6))

    # Plot each DataFrame's precision on the same plot
    for i, df in enumerate(dataframes):
        epochs = df['epoch']
        precision = df['metrics/precision(B)']
        adjusted_precision = precision * 10 + 0.4
        plt.plot(epochs, precision, marker='o', linestyle='-', label=labels[i])

    plt.title('Precision per Epoch (Comparison)')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_recall(dataframes, labels):
    """
    Plot recall vs. epoch for multiple DataFrames on the same plot.
    Args:
    - dataframes: List of DataFrames, each containing 'epoch' and 'metrics/recall(B)' columns.
    - labels: List of labels for each DataFrame, to use in the plot legend.
    """
    plt.figure(figsize=(8, 6))

    # Plot each DataFrame's recall on the same plot
    for i, df in enumerate(dataframes):
        epochs = df['epoch']
        recall = df['metrics/recall(B)']
        plt.plot(epochs, recall, marker='o', linestyle='-', label=labels[i])

    plt.title('Recall per Epoch (Comparison)')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_map50(dataframes, labels):
    """
    Plot mAP50 vs. epoch for multiple DataFrames on the same plot.
    Args:
    - dataframes: List of DataFrames, each containing 'epoch' and 'metrics/mAP50(B)' columns.
    - labels: List of labels for each DataFrame, to use in the plot legend.
    """
    plt.figure(figsize=(8, 6))

    # Plot each DataFrame's mAP50 on the same plot
    for i, df in enumerate(dataframes):
        epochs = df['epoch']
        map50 = df['metrics/mAP50(B)']
        plt.plot(epochs, map50, marker='o', linestyle='-', label=labels[i])

    plt.title('mAP50 per Epoch (Comparison)')
    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def process_directory(directory_path):
    """
    Load multiple CSV files from a directory, clean them, and plot precision, recall, and mAP50 for each.
    Args:
    - directory_path: Path to the directory containing CSV files.
    """
    dataframes = []
    labels = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}...")

            # Process each CSV file and store the DataFrame
            df = process_csv(file_path, return_df=True)
            dataframes.append(df)
            labels.append(filename)  # Use the filename as label for the plot

    # Plot the combined mAP50 for all experiments
    plot_map50(dataframes, labels)
    plot_precision(dataframes, labels)
    plot_recall(dataframes, labels)


def process_csv(file_path, return_df=False):
    """
    Load a CSV file, clean it, and optionally return the DataFrame.
    Args:
    - file_path: Path to the CSV file.
    - return_df: Whether to return the DataFrame after processing.
    """
    # Load CSV
    df = pd.read_csv(file_path)
    # Remove extra spaces from columns
    df.columns = df.columns.str.strip()
    # Filter every second row (remove secondary rows for each epoch)
    filtered_df = df.iloc[::2].reset_index(drop=True)

    if return_df:
        return filtered_df

if __name__ == '__main__':
    # Replace 'path_to_directory' with the actual path to your directory containing CSV files
    process_directory(r'D:\yolopm-all\YOLOv8-multi-task-paving\xhwTools\test\result')