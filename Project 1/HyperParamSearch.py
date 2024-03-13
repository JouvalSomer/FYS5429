import os
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
import torch
import pandas as pd
from Dataset import dataloader
from Model import HydrologyLSTM
import json


def hyperparameter_search(config, abs_path, epochs, cpu_cores, split_ratios):

    def train_model(config):
        data = pd.read_csv(abs_path, delimiter='\t')
        input_size = data.shape[1] - 2

        dataloaders = dataloader(data, lookback=config["lookback"], split_ratios=split_ratios, batch_size=config["batch_size"])
        model = HydrologyLSTM(input_size=input_size, hidden_size=config["hidden_size"], num_layers=config["num_layers"], drop_out=config["drop_out"])
        model.fit(dataloaders, epochs=epochs, lr=config["lr"], ray_tune=True)

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=100, # Maximum number of training iterations (epochs) for any given trial
        grace_period=2, # A trial must run for at least 10 iterations before it can be stopped early.
        reduction_factor=2 #  Determines how aggressively trials are stopped. The algorithm regularly halves the number of trials
    )

    reporter = CLIReporter(metric_columns=['loss', 'training_iteration'])

    ray.init()

    result = tune.run(
        train_model,
        resources_per_trial={'cpu': cpu_cores, 'gpu': 1},
        config=config,
        num_samples=10,     #  Number of times the training function will be executed with randomly 
        scheduler=scheduler,
        progress_reporter=reporter
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    best_trial_config = best_trial.config
    best_trial_metrics = best_trial.last_result

    print(f"Best trial config: {best_trial_config}")
    print(f"Best trial final validation loss: {best_trial_metrics['loss']}")

    best_trial_summary = {"config": best_trial_config, "metrics": best_trial_metrics}

    with open("best_trial_summary.json", "w") as f:
        json.dump(best_trial_summary, f, indent=4)


    return best_trial_config