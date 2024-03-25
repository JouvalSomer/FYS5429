from HyperParamSearch import hyperparameter_search
from ray import tune
import os
import pandas as pd
from Dataset import dataloader
from Model import HydrologyLSTM
import matplotlib.pyplot as plt
import torch


# from GeneratePlots import Plots
# from argparse import ArgumentParser
# import matplotlib.pyplot as plt

def run_hyperparameter_search(abs_path, epochs, num_samples, max_t, grace_period, cpu_cores):
    config = {
        "hidden_size": tune.choice([100, 200, 300]),
        "num_layers": tune.choice([1]),
        "drop_out": tune.choice([0.2, 0.3, 0.4]),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "lookback": tune.choice([270, 300, 365, 400])
    }

    return hyperparameter_search(config, abs_path, epochs, cpu_cores, split_ratios, num_samples, max_t, grace_period)


def run_model(best_trial_config, epochs, abs_path):
    lookback = best_trial_config["lookback"]
    batch_size = best_trial_config["batch_size"]
    hidden_size = best_trial_config["hidden_size"]
    num_layers = best_trial_config["num_layers"]
    drop_out = best_trial_config["drop_out"]
    lr = best_trial_config["lr"]

    data = pd.read_csv(abs_path, delimiter='\t')
    dataloaders = dataloader(data, lookback, split_ratios, batch_size)

    input_size = data.shape[1] - 2  # The number of features

    model = HydrologyLSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, drop_out=drop_out)

    model.fit(dataloaders,
              epochs=epochs,
              lr=lr,
              store_data=False,
              PATH=None,
              n_print=10)

    avg_loss, y_hat_test_set, y_test_set, cell_states_test = model.predict(
        dataloaders['test'], ray_tune=False)
    avg_loss, y_hat_validation_set, y_validation_set, cell_states_val = model.predict(
        dataloaders['validate'], ray_tune=False)
    avg_loss, y_hat_train_set, y_train_set, cell_states_train = model.predict(
        dataloaders['train_not_shuffled'], ray_tune=False)

    loss = (model.train_loss, model.validation_loss)
    train = (y_hat_train_set, y_train_set)
    validation = (y_hat_validation_set, y_validation_set)
    test = (y_hat_test_set, y_test_set)

    return loss, train, validation, test


def run_plots(loss, train, validation, test):
    y_hat_train_set, y_train_set = train
    y_hat_validation_set, y_validation_set = validation
    train_loss, validation_loss = loss
    y_hat_test_set, y_test_set = test

    y_hat_train_set_np = torch.cat(y_hat_train_set, dim=0).detach().numpy()
    y_train_set_np = torch.cat(y_train_set, dim=0).detach().numpy()

    y_hat_validation_set_np = torch.cat(
        y_hat_validation_set, dim=0).detach().numpy()
    y_validation_set_np = torch.cat(y_validation_set, dim=0).detach().numpy()

    plt.figure(figsize=(10, 6))
    plt.plot(y_hat_train_set_np, label='LSTM Prediction (train)', alpha=0.7)
    plt.plot(y_train_set_np, label='Qobs (data, train)', alpha=0.7)
    plt.title('LSTM Predictions vs Observed Data')
    plt.xlabel('Days')
    plt.ylabel('Discharge')
    plt.legend()
    plt.savefig('LSTM_Predictions_vs_Observed_Data_new.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_hat_validation_set_np,
             label='LSTM Prediction (validation)', alpha=0.7)
    plt.plot(y_validation_set_np, label='Qobs (data, validation)', alpha=0.7)
    plt.title('LSTM Predictions vs Observed Data')
    plt.xlabel('Days')
    plt.ylabel('Discharge')
    plt.legend()
    plt.savefig('LSTM_Predictions_vs_Observed_Data_new.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_loss)), train_loss,
             label='Training Loss', alpha=0.7)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_Loss_vs_Epochs_new.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(validation_loss)), validation_loss,
             label='Validation Loss', alpha=0.7)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Training_Loss_vs_Epochs_new.png')
    plt.show()


if __name__ == "__main__":

    abs_path = os.path.abspath("ptq.txt")

    train, validation, test = 0.7, 0.2, 0.1
    split_ratios = (train, validation, test)

    # Number of epochs in the model training (epochs <= max_t)
    epochs = 100
    num_samples = 20    # Number of random samples to be tested
    # Maximum number of training iterations (epochs) for any given trial
    max_t = 100
    # A trial must run for at least grace_period iterations before it can be stopped early.
    grace_period = 2

    cpu_cores = 4       # Number of cpu cores to be used in the hyperparameter search

    best_trial_config = run_hyperparameter_search(
        abs_path, epochs, num_samples, max_t, grace_period, cpu_cores)

    loss, train, validation, test = run_model(
        best_trial_config, epochs, abs_path)

    run_plots(loss, train, validation, test)
