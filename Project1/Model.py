import torch
import torch.nn as nn
from ray import train
from ray import tune


class LSTMBlock(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 T: int = 400,
                 ) -> None:

        self.T = T
        # self.LSTMcells = nn.ModuleList(nn.LSTMCell(
        #    input_size=input_size, hidden_size=hidden_size) for _ in range(T))
        self.LSTMCell = nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_size)
        self.cell_storage = []
        pass

    def forward(self, x):

        # x = [batch, features, T]

        h = None
        c = None

        for t in range(self.T):
            h, c = self.LSTMCell(x[:, :, t], h, c)
            self.storage.append(c)


class HydrologyLSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 drop_out: float = 0.4
                 ):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__()
        self.train_loss = []
        self.validation_loss = []

        self.y_hat_validation_set = []
        self.y_validation_set = []

        self.y_hat_train_set = []
        self.y_train_set = []

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.l_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(drop_out)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

        self.initialize_weights()

    def forward(self, x):

        x = self.l_norm(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)[:, -1, :]
        x = self.fc(x)
        x = self.relu(x)

        return x

    def forward_predict(self, x):

        x = self.l_norm(x)
        x, (h, c) = self.lstm(x)
        x = self.dropout(x)[:, -1, :]
        x = self.fc(x)
        x = self.relu(x)

        return x, c

    def loss_function(self, y_pred, y):
        mean_true = torch.mean(y)
        numerator = torch.sum(torch.square(y - y_pred))
        denominator = torch.sum(torch.square(y - mean_true))
        nse_loss = numerator / denominator
        return nse_loss

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:  # Bias
                param.data.fill_(0)

    def fit(self,
            dataloaders,
            epochs: int,
            lr: float,
            store_data=False,
            PATH: str = None,
            n_print: int = 10,
            ray_tune=False
            ) -> None:

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.train()

        for e in range(epochs):

            running_loss = 0

            for x_batch, y_batch in dataloaders['train']:

                optimizer.zero_grad()
                pred = self.forward(x_batch.to(self.device))

                loss = loss_fn(pred, y_batch.to(self.device))

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            e_loss = running_loss / len(dataloaders['train'].dataset)
            self.train_loss.append(e_loss)

            if e % n_print == 0:
                print(f"Epoch: {e} || Loss: {e_loss}")

            if ray_tune:
                val_loss, y_hat_set, y_set = self.predict(
                    dataloaders['validate'], ray_tune)
                train.report({'loss': val_loss})

        if store_data:
            torch.save(self, PATH)

    def predict(self, dataloader, ray_tune) -> tuple[float, list, list, list]:

        y_hat_set = []
        y_set = []
        total_loss = 0
        loss_fn = nn.MSELoss()

        self.eval()
        with torch.no_grad():
            for x_batch, y_batch in dataloader:
                pred = self.forward(x_batch.to(self.device))

                y_hat_set.append(pred)
                y_set.append(y_batch)

                total_loss += loss_fn(pred,
                                      y_batch.to(self.device)).item()

        avg_loss = total_loss / len(dataloader.dataset)

        if not ray_tune:
            self.validation_loss.append(avg_loss)

        print(f'Validation Loss: {avg_loss}')

        return avg_loss, y_hat_set, y_set
