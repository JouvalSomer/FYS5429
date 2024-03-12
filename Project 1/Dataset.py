import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class Data(Dataset):
    def __init__(self, features, targets):
        """
        Initialize the dataset with features and targets.
        
        Parameters:
        - features: A NumPy array of input features.
        - targets: A NumPy array of target values.
        """
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        """
        Get the features and target for a given index.
        
        Parameters:
        - idx: The index of the data point.
        
        Returns:
        A tuple containing the feature tensor and target tensor.
        """
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target

def dataloader(data, lookback, split_ratios, batch_size):
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

    # Calculate indices for splits
    total_samples = len(data)
    train_end = int(total_samples * split_ratios[0])
    validate_end = train_end + int(total_samples * split_ratios[1])

    def generate_sequences(features, targets, start_idx, end_idx, lookback):
        X, Y = [], []
        for i in range(start_idx + lookback, end_idx):
            X.append(features[i-lookback:i])
            Y.append(targets[i])

        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1)

    # Generate sequences
    X_train, Y_train = generate_sequences(features, targets, 0, train_end, lookback)
    X_validate, Y_validate = generate_sequences(features, targets, train_end, validate_end, lookback)
    X_test, Y_test = generate_sequences(features, targets, validate_end, total_samples, lookback)

    # Create datasets
    train_dataset = Data(X_train, Y_train)
    validate_dataset = Data(X_validate, Y_validate)
    test_dataset = Data(X_test, Y_test)

    # Create DataLoaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'validate': DataLoader(validate_dataset, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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
