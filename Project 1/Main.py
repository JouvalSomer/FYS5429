import torch
import pandas as pd
from Dataset import Data
from Model import HydrologyLSTM
from GeneratePlots import Plots
from argparse import ArgumentParser




if __name__ == "__main__":

    data = pd.read_csv("ptq.txt", delimiter='\t')
    epochs = 300
    lookback = 360

    train = 0.7
    test = 0.2
    validation = 0.1

    myData = Data(data, lookback = lookback, split = (train, test, validation))

    train = myData.train_DataLoader
    test = myData.test_DataLoader
    val = myData.validation_DataLoader
    
    model = HydrologyLSTM()

    model.fit(train, 
            epochs = 100, 
            lr  = 0.001, 
            store_data = False, 
            PATH = None, 
            n_print = 10)

    pred_val_np = torch.cat(model.pred_validation_set, dim=0).numpy()
    y_val_np = torch.cat(model.y_validation_set, dim=0).numpy()