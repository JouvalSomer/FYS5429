from HyperParamSearch import hyperparameter_search
from ray import tune
import os
import pandas as pd
from Dataset import dataloader
from Model import HydrologyLSTM


# from GeneratePlots import Plots
# from argparse import ArgumentParser
# import matplotlib.pyplot as plt

def run_hyperparameter_search(abs_path):
    config = {
        "hidden_size": tune.choice([100, 200, 300]),
        "num_layers": tune.choice([1, 2, 3]),
        "drop_out": tune.uniform(0.2, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "lookback": tune.choice([200, 250, 300, 365])
    }

    epochs = 10
    cpu_cores = 4

    return hyperparameter_search(config, abs_path, epochs, cpu_cores, split_ratios)

def run_model(best_trial_config, epochs, abs_path):
     
     lookback = best_trial_config["lookback"]
     batch_size = best_trial_config["batch_size"]
     hidden_size = best_trial_config["hidden_size"]
     num_layers = best_trial_config["num_layers"]
     drop_out = best_trial_config["drop_out"]
     lr = best_trial_config["lr"]
     
     data = pd.read_csv(abs_path, delimiter='\t')
     dataloaders = dataloader(data, lookback, split_ratios, batch_size)
     
     input_size = data.shape[1] - 2 # The number of features
     
     model = HydrologyLSTM(input_size = input_size, hidden_size = hidden_size, 
                          num_layers = num_layers, drop_out = drop_out)
     
     model.fit(dataloaders['train'], 
        epochs = epochs, 
        lr  = lr, 
        store_data = False, 
        PATH = None, 
        n_print = 10)
     
     model.predict(dataloaders['validate'])
     

if __name__ == "__main__":

        abs_path = os.path.abspath("ptq.txt")


        train, validation, test = 0.7, 0.2, 0.1
        split_ratios = (train, validation, test)

        epochs = 100

        best_trial_config = run_hyperparameter_search(abs_path)
        run_model(best_trial_config, epochs, abs_path)




 