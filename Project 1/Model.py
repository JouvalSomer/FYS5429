import torch
import torch.nn as nn
from ray import train
from ray import tune


class HydrologyLSTM(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, drop_out: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        self.epoch_loss = []
        self.pred_validation_set = []
        self.y_validation_set = []

    def forward(self, x):

        x, _ = self.lstm(x)
        x = self.dropout(x)[:, -1, :]
        x = self.fc(x)
        x = self.relu(x)

        return x

    def loss_function(self, y_pred, y):
        mean_true = torch.mean(y)
        numerator = torch.sum(torch.square(y - y_pred))
        denominator = torch.sum(torch.square(y - mean_true))        
        nse_loss = numerator / denominator
        return nse_loss


    def fit(self, dataloaders, epochs: int, lr: float = 0.001, store_data = False, PATH: str = None, n_print: int = 10, ray_tune = False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        for e in range(epochs):

            running_loss = 0

            for x_batch, y_batch in dataloaders['train']:

                optimizer.zero_grad()
                pred = self.forward(x_batch)
                loss = self.loss_function(pred, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            e_loss = running_loss / len(dataloaders['train'])
            self.epoch_loss.append(e_loss)
        
            if e % n_print == 0:
                print(f"Epoch: {e} || Loss: {e_loss}")

            if ray_tune:
                val_loss = self.predict(dataloaders['validate'])
                train.report({'loss': val_loss}) 

        if store_data:
            torch.save(self, PATH)


    def predict(self, dataloader):
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                pred = self.forward(x_batch)
                self.pred_validation_set.append(pred)
                self.y_validation_set.append(y_batch)

                total_loss += self.loss_function(pred, y_batch).item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f'Validation Loss: {avg_loss}')

        return avg_loss

