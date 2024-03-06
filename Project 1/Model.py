import torch
import torch.nn as nn
import numpy as np


class HydrologyLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.fc(x)

        return x

    def loss_function(self, y_pred, y):
        # TODO:
        # Define loss function
        pass

    def fit(self, dataloader, epochs: int = 100, lr: float = 0.001, store_data=False, PATH: str = None):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        for e in range(epochs):

            for x_batch, y_batch in dataloader:

                optimizer.zero_grad()
                pred = self.forward(x_batch)
                loss = self.loss_function(pred, y_batch)
                loss.backward()
                optimizer.step()

                # TODO:
                # Add storing of loss and other parameters

        if store_data:
            torch.save(self, PATH)

    def predict(self, dataloader):

        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in dataloader:

                pred = self.forward(x_batch)
                loss = self.loss_fn(pred, y_batch)
