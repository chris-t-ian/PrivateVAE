import optuna
import torch
import pandas as pd

from p_vqvae.dataloader import DataSet, get_train_val_loader
from p_vqvae.neural_spline_flow import NSF

data_mode = "raw"  # TODO: implement hyperparameter search for raw and for synthetic data

data_kwargs = {
    "root": "/home/chrsch/P_VQVAE/data/ATLAS_2",
    "cache_path": '/home/chrsch/P_VQVAE/data/cache/',
    "downsample": 4,  # change to 1
    "normalize": 1,
    "crop": ((8, 9), (12, 13), (0, 9)),
    "padding": ((1, 2), (0, 0), (1, 2)),
}
train_loader_kwargs = {
    "num_workers": 2,
    "split_ratio": 0.9,
}
model_params = {
    "model_path": "/home/chrsch/P_VQVAE/model_outputs/lira",
    "device": "cuda",
    "num_steps": 100000,
    "eval_interval": 20,  # after how many steps evaluate on validation set
    "early_stopping": 3,  # after how many evaluation steps stop the training
    "steps_per_level": 10,
    "levels": 2,  # increase for non-downsampled dataset
    "multi_scale": True,
    "actnorm": False,
}
_spline_params = {
        'tail_bound': 1.,
        'min_bin_width': 1e-3,
        'min_bin_height': 1e-3,
        'min_derivative': 1e-3,
        'apply_unconditional_transform': False
}
optimization = {
    "augment_flag": True,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "cosine_annealing": False,
    'num_bins': 4,
    "hidden_channels": 64
}
coupling_transform = {
    "coupling_layer_type": 'rational_quadratic_spline',
    "use_resnet": False,
    "num_res_blocks": 5,  # If using resnet
    "resnet_batchnorm": True,
    "dropout_prob": 0.,
}
nsf_kwargs = {**model_params, **coupling_transform, "spline_parameters": _spline_params}

hyperparameter_log = "/home/chrsch/P_VQVAE/model_outputs/hyperparameters_search_nsf.csv"


def objective_vqvae(trial):
    # Define the hyperparameter search space for the VQVAE
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    augment_flag = trial.suggest_categorical("augment_flag", [True, False])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    cosine_annealing = trial.suggest_categorical("cosine_annealing", [True, False])
    hidden_channels = trial.suggest_categorical("hidden_channels", [64, 128, 256])
    num_bins = trial.suggest_categorical("num_bins", [4, 6, 8])

    if data_mode == "raw":
        dataset = DataSet("half", **data_kwargs)
    else:
        raise NotImplementedError

    train_loader, val_loader = get_train_val_loader(dataset, batch_size=batch_size, augment_flag=augment_flag,
                                                    **train_loader_kwargs)

    nsf = NSF(
        train_loader,
        val_loader,
        learning_rate=learning_rate,
        cosine_annealing=cosine_annealing,
        hidden_channels=hidden_channels,
        num_bins=num_bins,
        **nsf_kwargs
    )
    best_val_log_prob = nsf.train()

    return best_val_log_prob


if __name__ == '__main__':
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_vqvae, n_trials=10)

    # Best hyperparameters
    print("Best trial:")
    print(f"  Validation Loss: {study.trial.value}")
    print("  Best hyperparameters: ", study.trial.params)

    # save trials to log
    results = []

    trial_info = {'Trial': f"-1", 'Validation Loss': study.best_trial}.update(study.trial.params)
    results.append(trial_info)
    for t in study.trials:
        trial_info = {'Trial': t.number, 'Validation Loss': t.value}
        trial_info.update(t.params)
        results.append(trial_info)

    df_results = pd.DataFrame(results)

    # Save to CSV
    df_results.to_csv(hyperparameter_log, index=False)
