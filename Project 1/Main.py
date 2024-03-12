import torch
import pandas as pd
from Dataset import dataloader
from Model import HydrologyLSTM
from GeneratePlots import Plots
from argparse import ArgumentParser
import matplotlib.pyplot as plt


if __name__ == "__main__":

    data = pd.read_csv("ptq.txt", delimiter='\t')

    lookback = 360

    train = 0.7
    validation = 0.2
    test = 0.1
    split_ratios = (train, validation, test)

    batch_size = 32

    dataloaders = dataloader(data, lookback, split_ratios, batch_size)

    input_size = data.shape[1] - 2 # The number of features

    model = HydrologyLSTM(input_size = input_size, hidden_size = 200, 
                          num_layers = 1, drop_out = 0.4)

    epochs = 50
    lr = 0.001

    model.fit(dataloaders['train'], 
            epochs = epochs, 
            lr  = lr, 
            store_data = False, 
            PATH = None, 
            n_print = 10)

    model.predict(dataloaders['validate'])

    pred_val_np = torch.cat(model.pred_validation_set, dim=0).numpy()
    y_val_np = torch.cat(model.y_validation_set, dim=0).numpy()


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
    plt.plot(range(epochs), model.epoch_loss, label='Training Loss', alpha=0.7)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_Loss_vs_Epochs_new.png')
    plt.show()