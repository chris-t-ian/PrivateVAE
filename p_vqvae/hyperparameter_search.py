import optuna
import torch
import pandas as pd

from p_vqvae.dataloader import DataSet, get_train_val_loader
from p_vqvae.networks import VQ_VAE
from p_vqvae.visualise import plot_generated_images, plot_reconstructions
from monai.utils import set_determinism
from p_vqvae.metrics import calculate_and_save_all_metrics

epochs = 150
down_sampling = 4
root = "/home/chrsch/P_VQVAE/data/ATLAS_2"
cache_path = '/home/chrsch/P_VQVAE/data/cache/'
model_paths = "/home/chrsch/P_VQVAE/model_outputs/model_weights"
metrics_path = "/home/chrsch/P_VQVAE/model_outputs"
hyperparameter_log = "/home/chrsch/P_VQVAE/model_outputs/hyperparameters_search.csv"


def objective_vqvae(trial):
    # Define the hyperparameter search space
    augment_flag = trial.suggest_categorical("augmentation", [True, False])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    learning_rate = trial.suggest_loguniform("learning_rate", 5e-5, 1e-3)
    num_embeddings = trial.suggest_int("num_embeddings", 128, 1048, step=128)
    embedding_dim = trial.suggest_int("embedding_dim", 32, 64, step=16)
    num_res_layers = trial.suggest_int("num_res_layers", 2, 3, step=1)

    if num_res_layers == 2:
        num_channels = (
            trial.suggest_int("num_channels_layer1", 256, 384, step=64),
            trial.suggest_int("num_channels_layer2", 256, 512, step=64),
        )
        downsample_parameters = ((2, 4, 1, 1), (2, 4, 1, 1))
        upsample_parameters = ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0))
    elif num_res_layers == 3:
        num_channels = (
            trial.suggest_int("num_channels_layer1", 128, 256, step=64),
            trial.suggest_int("num_channels_layer2", 128, 256, step=64),
            trial.suggest_int("num_channels_layer2", 256, 512, step=64),
        )
        downsample_parameters = ((2, 4, 1, 1), (2, 4, 1, 1), (2, 4, 1, 1))
        upsample_parameters = ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0), (2, 4, 1, 1, 0))
    else:
        raise NotImplementedError

    dataset = DataSet("full",
                      root,
                      cache_path,
                      downsample=down_sampling,
                      normalize=True,
                      crop=((8, 9), (12, 13), (0, 9)),
                      padding=((1, 2), (0, 0), (1, 2)))
    train_loader, val_loader = get_train_val_loader(dataset, batch_size=batch_size, split_ratio=0.875, num_workers=4,
                                                    augment_flag=augment_flag)

    vq_vae_model = VQ_VAE(train_loader, val_loader,
                          n_epochs=epochs,
                          multiple_devices=False,
                          lr=learning_rate,
                          num_embeddings=num_embeddings,
                          embedding_dim=embedding_dim,
                          channels=num_channels,
                          num_res_layers=num_channels,
                          downsample_parameters=downsample_parameters,
                          upsample_parameters=upsample_parameters,
                          use_checkpointing=True,
                          val_interval=1)
    val_loss = vq_vae_model.train()

    return val_loss


if __name__ == '__main__':
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_vqvae, n_trials=50)

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
