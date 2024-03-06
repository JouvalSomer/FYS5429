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
            split - How to split into training, validation and test sets
        """
        if not isinstance(lookback, int):
            print(f"Needs to be of type int, not {type(lookback)}")
            raise TypeError

        if not isinstance(data, pd.DataFrame):
            print(f"Needs to be a pandas DataFrame object, not {type(data)}")
            raise TypeError

        super().__init__()
        self.data_shape = data.shape

        self.X = []
        self.Y = []
        self.data = data
        self.lookback = lookback

        # Extracting features and target values from DataFrame
        self.timeseries = np.zeros(
            (self.data_shape[0], self.data_shape[1] - 1))

        # TODO:
        #####################
        # Add train test split
        ####################

        keys = list(data.keys())

        if keys[0] not in ["Date", "date", "Dates", "dates"]:
            print("Please use a DataFrame with the dates in the first columns")
            raise ValueError

        self.num_features = len(keys)

        for key in range(self.num_features-1):
            self.timeseries[:, key] = data[keys[key]].values.astype(np.float32)

        # Creating feature and target arrays
        for i in range(self.data_shape[0] - self.lookback):
            feature = self.timeseries[i:i+self.lookback, :self.num_features-2]
            target = self.timeseries[i+1:i +
                                     self.lookback+1, self.num_features-2:]
            self.X.append(feature)
            self.Y.append(target)

    def __len__(self) -> int:
        return self.data_shape[0] - self.lookback

    def __getitem__(self, idx) -> tuple[torch.tensor, torch.tensor]:
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])


if __name__ == "__main__":

    data = pd.read_csv("ptq.txt", delimiter='\t')
    myData = Data(data, 365)

    print(myData[0])
