
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# How to install Ray Tune: pip install "ray[tune]"
from ray import tune
from ray import train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

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
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
    

class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=config.get('hidden_size', 256),
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(config.get('dropout_rate', 0.4))  
        self.fc = nn.Linear(config.get('hidden_size', 256), 1) 
        self.relu = nn.ReLU() 

    def forward(self, x):
        x, _ = self.lstm(x)  
        x = self.dropout(x[:, -1, :])  
        x = self.fc(x)
        x = self.relu(x)
        return x


abs_path = os.path.abspath("ptq.txt")

def train_model(config):
    data = pd.read_csv(abs_path, delimiter='\t')
    timeseries = np.column_stack((data["Prec."].values, data["Temp"].values, data["Qobs"].values)).astype(np.float32)

    train_size = int(len(timeseries) * 0.8)
    train_data = timeseries[:train_size, :]
    val_data = timeseries[train_size:, :]

    train_dataset = Data(train_data, 365)
    val_dataset = Data(val_data, 365)

    train_dataloader = DataLoader(train_dataset, batch_size=int(config["batch_size"]), shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=int(config["batch_size"]), shuffle=False)

    # Initialize model
    model = Model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Checkpoint handling
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
            model_state, optimizer_state = torch.load(checkpoint_path)
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    # Training loop
    for epoch in range(10):  # Number of epochs
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += loss_fn(pred, y).item()
        val_loss /= len(val_dataloader)

        # Reporting metrics
        train.report({'loss': val_loss})


if __name__ == '__main__':
    # Example search:
    config = {
        'hidden_size': tune.choice([128, 256, 512]),
        'dropout_rate': tune.choice([0.2, 0.25, 0.3, 0.35, 0.4]),
        'lr': tune.loguniform(1e-4, 1e-2),
        'batch_size': tune.choice([16, 32, 64])
    }

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=100,
        grace_period=10,
        reduction_factor=2)
    
    reporter = CLIReporter(metric_columns=['loss', 'training_iteration'])
    
    result = tune.run(
        train_model,
        resources_per_trial={'cpu': 1, 'gpu': 1},
        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter
    )
