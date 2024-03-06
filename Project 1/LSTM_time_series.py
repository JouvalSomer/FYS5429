import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Data(Dataset):
    def __init__(self, data, lookback) -> None:
        super().__init__()
        self.X = []
        self.y = []

        for i in range(len(data) - lookback):
            feature = data[i:i+lookback, :2]
            target = data[i+lookback, 2]
            self.X.append(feature)
            self.y.append(target)

        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 1)
         
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        p = (torch.tensor(self.X[idx]), torch.tensor(self.y[idx]))
        return p    
    

class Model(nn.Module):

    def __init__(self, dropout_rate=0.4) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=256, 
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)  
        self.fc = nn.Linear(256, 1) 
        self.relu = nn.ReLU() 

    def forward(self, x):
        x, _ = self.lstm(x)  
        x = self.dropout(x[:, -1, :])  
        x = self.fc(x)
        x = self.relu(x)
        return x
    

def train_model(dataloader, model, loss_fn, optimizer, epochs):
    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_loss = total_loss / len(dataloader)
        epoch_losses.append(epoch_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {epoch_loss}')
    return epoch_losses


def evaluate_model(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    pred_val = []
    y_val = []
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_val.append(pred)
            y_val.append(y)
            total_loss += loss_fn(pred, y).item()
            
    print(f'Validation Loss: {total_loss / len(dataloader)}') # dataloader.dataset
    return pred_val, y_val



if __name__ == "__main__":
    data = pd.read_csv("ptq.txt", delimiter='\t')
    timeseries = np.column_stack((data["Prec."].values, data["Temp"].values, data["Qobs"].values)).astype(np.float32)

    train_size = int(len(timeseries) * 0.8)
    train_data = timeseries[ : train_size, :]
    val_data = timeseries[train_size :, :]


    train_dataset = Data(train_data, 365)
    val_dataset = Data(val_data, 365)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = Model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 300
    epoch_losses = train_model(train_dataloader, model, loss_fn, optimizer, epochs=epochs)

    pred_val, y_val = evaluate_model(val_dataloader, model, loss_fn)

    pred_val_np = torch.cat(pred_val, dim=0).numpy()
    y_val_np = torch.cat(y_val, dim=0).numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(pred_val_np, label='LSRM Prediction', alpha=0.7)
    plt.plot(y_val_np, label='Qobs (data)', alpha=0.7)
    plt.title('LSTM Predictions vs Observed Data')
    plt.xlabel('Days')
    plt.ylabel('Discharge')
    plt.legend()
    plt.savefig('LSTM_Predictions_vs_Observed_Data_new.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), epoch_losses, label='Training Loss', alpha=0.7)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_Loss_vs_Epochs_new.png')
    plt.show()