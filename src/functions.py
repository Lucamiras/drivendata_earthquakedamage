import pandas as pd

def load_data():
    """Load data from file."""
    train_values = pd.read_csv('data/train_values.csv')
    train_labels = pd.read_csv('data/train_labels.csv')
    test_values = pd.read_csv('data/test_values.csv')
    return train_values, train_labels, test_values