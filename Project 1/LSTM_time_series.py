import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


class Data(Dataset):

    def __init__(self, data, lookback) -> None:
        super().__init__()
        self.X = []
        self.y = []
        self.data = data
        self.lookback = lookback

        for i in range(len(data) - lookback):
            feature = data[i:i+lookback, :2]
            target = data[i+1:i+lookback+1, 2:]
            self.X.append(feature)
            self.y.append(target)

        # print(len(self.X), self.X[0].shape)
        # print(len(self.y), self.y[0].shape)

    def __len__(self):
        return self.data.shape[0] - self.lookback

    def __getitem__(self, idx):
        p = (torch.tensor(self.X[idx]), torch.tensor(self.y[idx]))
        # print(p[1].shape)
        return p
        # return (torch.tensor(self.X[idx]), torch.tensor(self.y[idx]))


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=50,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.fc2 = nn.Linear(50, 1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)

        return x


def fit(dataloader, model, loss_fn, lr=0.001, epochs=100, n_print=100):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_loss = []

    for e in range(epochs):

        for x_batch, y_batch in dataloader:

            # print(f"before: {x_batch.shape}")
            pred = model.forward(x_batch)
            # print(f"After: {pred.shape}")
            loss = loss_fn(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        if e % n_print == 0:
            print("Mean loss per epoch")
            print(f"Loss:{np.mean(total_loss)} || Epoch: {e}")


def accuracy(dataloader, model, loss_fn):

    pred_val = []
    y_val = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:

            pred = model.forward(x_batch)
            loss = loss_fn(pred, y_batch)

            y_val.append(y_batch)
            pred_val.append(pred)

    return pred_val, y_val


if __name__ == "__main__":

    data = pd.read_csv("ptq.txt", delimiter='\t')
    timeseries = np.zeros((data.shape[0], 3), dtype=np.float32)
    # print(data[["Prec."]].values)
    timeseries[:, 0] = data["Prec."].values.astype(np.float32)
    timeseries[:, 1] = data["Temp"].values.astype(np.float32)
    timeseries[:, 2] = data["Qobs"].values.astype(np.float32)

    train = timeseries[:2000, :]
    val = timeseries[2000:3000, :]

    train_data = Data(train, 365)
    val_data = Data(val, 365)

    train_dataloader = DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)
    loss_fn = nn.MSELoss()

    model = Model()

    # print(train_dataloader[0].shape)

    fit(train_dataloader, model, loss_fn, 0.01, 100, 10)
    pred_val, y_val = accuracy(val_dataloader, model, loss_fn)
    print(pred_val[0])
    print(pred_val[0].shape)
    print(pred_val)
    print(y_val)

    plt.scatter(np.linspace(0, 5, 365), pred_val[0])
    plt.show()

    """
    dates = [datetime.strptime(f"{date}", "%Y%m%d") for date in data["date"]]

    plt.plot_date(dates, data["Prec."], xdate=True)
    plt.plot_date(dates, data["Temp"], xdate=True)
    plt.plot_date(dates, data["Qobs"], xdate=True)
    plt.show()
    """
