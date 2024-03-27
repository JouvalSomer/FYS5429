import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd


def dataloader(data, lookback, split_ratios, batch_size, num_workers=4, pin_memory=False):
    """
    Prepares training, validation, and test datasets and their respective DataLoaders using the custom Data class.

    Parameters:
    - data: A pandas DataFrame containing the time series data.
    - lookback: The number of time steps to use for each input sequence.
    - split_ratios: A tuple of three elements indicating the split ratios for training, validation, and test sets.
    - batch_size: The batch size for the DataLoaders.

    Returns:
    A dictionary containing 'train', 'validate', and 'test' DataLoaders.
    """
    features = data.iloc[:, 1:-1].values
    targets = data.iloc[:, -1].values

    # features = data[:, 1:-1]
    # targets = data[:, -1]

    # Calculate indices for splits
    total_samples = len(data)
    train_end = int(total_samples * split_ratios[0])
    validate_end = train_end + int(total_samples * split_ratios[1])

    # Generate sequences

    def generate_sequences(features, targets, start_idx, end_idx, lookback):
        X, Y = [], []
        for i in range(start_idx + lookback, end_idx):
            X.append(features[i-lookback:i])
            Y.append(targets[i])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1)

    X_train, Y_train = generate_sequences(
        features, targets, 0, train_end, lookback)

    X_validate, Y_validate = generate_sequences(
        features, targets, train_end, validate_end, lookback)

    X_test, Y_test = generate_sequences(
        features, targets, validate_end, total_samples, lookback)

    # Convert to PyTorch tensors and create datasets

    def dataset(x, y):
        return TensorDataset(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float))

    train_dataset = dataset(X_train, Y_train)
    validate_dataset = dataset(X_validate, Y_validate)
    test_dataset = dataset(X_test, Y_test)

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        'train_not_shuffled': DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        'validate': DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    }

    return dataloaders


if __name__ == "__main__":

    data = pd.read_csv("ptq.txt", delimiter='\t')

    lookback = 365

    train = 0.7
    validation = 0.2
    test = 0.1
    split_ratios = (train, validation, test)

    batch_size = 32

    dataloaders = dataloader(data, lookback, split_ratios, batch_size)
