import ray
from ray import tune
from ray.tune import Stopper
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import pandas as pd
import numpy as np
import json

from Dataset import dataloader
from Model import HydrologyLSTM

class NaNOrInfStopper(Stopper):
    def __call__(self, trial_id, result):
        """Returns True if the trial should be stopped."""
        return not np.isfinite(result["loss"])

    def stop_all(self):
        """Returns whether to stop all trials and prevent new ones from starting."""
        return False

def hyperparameter_search(config, abs_path, epochs, cpu_cores, split_ratios, num_samples=10, max_t=100, grace_period=10):

    def train_model(config):
        data = pd.read_csv(abs_path, delimiter='\t')
        input_size = data.shape[1] - 2

        dataloaders = dataloader(data, lookback=config["lookback"], split_ratios=split_ratios, batch_size=config["batch_size"])
        model = HydrologyLSTM(input_size=input_size, hidden_size=config["hidden_size"], num_layers=config["num_layers"], drop_out=config["drop_out"])
        model.fit(dataloaders, epochs=epochs, lr=config["lr"], ray_tune=True)

    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=max_t, # Maximum number of training iterations (epochs) for any given trial
        grace_period=grace_period, # A trial must run for at least grace_period iterations before it can be stopped early.
        reduction_factor=2 #  Determines how aggressively trials are stopped. The algorithm regularly halves the number of trials
    )

    reporter = CLIReporter(metric_columns=['loss', 'training_iteration'])

    ray.init()

    nan_or_inf_stopper = NaNOrInfStopper()

    result = tune.run(
        train_model,
        resources_per_trial={'cpu': cpu_cores, 'gpu': 1},
        config=config,
        num_samples=num_samples,     #  Number of random samples to be tested 
        scheduler=scheduler,
        progress_reporter=reporter,
        stop=nan_or_inf_stopper
    )

    best_trial = result.get_best_trial("loss", "min", "last")

    if best_trial is None:
        print("No successful trials.")
        return None

    best_trial_config = best_trial.config
    best_trial_metrics = best_trial.last_result

    print(f"Best trial config: {best_trial_config}")
    print(f"Best trial final validation loss: {best_trial_metrics['loss']}")

    best_trial_summary = {"config": best_trial_config, "metrics": best_trial_metrics}

    with open("best_trial_summary.json", "w") as f:
        json.dump(best_trial_summary, f, indent=4)


    return best_trial_config