import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class Data(Dataset):

    def __init__(self, data: pd.DataFrame, lookback: int, split: tuple[float, float, float]):
        """
        Creates pyTorch dataset from a pandas DataFrame. Discards the datetime values
        and extracts the remaining values, splitting them into features and targets.
        Datetimes are expected to be in first column of DataFrame.

        Parameters:
            data - A pandas DataFrame 
            lookback - An integer telling how many days the model will remember
            split - How to split into training, validation and test sets (fractions)
        """
        if not isinstance(lookback, int):
            print(f"Parameter 'lookback' needs to be of type int, not {type(lookback)}!")
            raise TypeError

        if not isinstance(data, pd.DataFrame):
            print(f"Parameter 'data' needs to be a pandas DataFrame object, not {type(data)}!")
            raise TypeError

        if sum(split) > 1:
            print(f"Parameter 'split' fractions must sum up to 1, current sum = {sum(split):.4f}!")
            raise ValueError

        super().__init__()
        self.data_shape = data.shape

        if data.keys()[0] not in ["Date", "date", "Dates", "dates"]:
            print("Please use a DataFrame with the dates in the first columns")
            raise ValueError

        
        self.dates = pd.to_datetime(data.iloc[:, 0])

        self.features = data.iloc[:, 1:-1].to_numpy(dtype=np.float32)
        self.targets = data.iloc[:, -1].to_numpy(dtype=np.float32)

        X, Y = [], []

        # Generate sequences for features and corresponding targets
        for i in range(len(self.features) - lookback):
            X.append(self.features[i:i + lookback])
            Y.append(self.targets[i + lookback])

        self.X = np.array(X, dtype=np.float32)
        self.Y = np.array(Y, dtype=np.float32).reshape(-1, 1)


        # Splitting the dataset into train, test and validation sets
        total_size = len(self.X)
        train_end = int(total_size * split[0])
        test_end = int(total_size * (split[0] + split[1]))

        self.train = (self.X[:train_end], self.Y[:train_end])
        self.test = (self.X[train_end:test_end], self.Y[train_end:test_end])
        self.validation = (self.X[test_end:], self.Y[test_end:])

        self.train_DataLoader = DataLoader(self.train, batch_size=32, shuffle=True)
        self.test_DataLoader = DataLoader(self.test, batch_size=32, shuffle=True)
        self.validation_DataLoader = DataLoader(self.validation, batch_size=32, shuffle=False)


    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

if __name__ == "__main__":

    data = pd.read_csv("ptq.txt", delimiter='\t')

    lookback = 360

    train = 0.7
    test = 0.2
    validation = 0.1

    myData = Data(data, lookback = lookback, split = (train, test, validation))

    print(f"Test set size: {len(myData.train[0])}")
    print(f"Test set size: {len(myData.test[0])}")
    print(f"Validation set size: {len(myData.validation[0])}")
