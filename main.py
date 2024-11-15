from PrivateVAE.p_vqvae.dataloader import RawDataSet, get_train_loader
from PrivateVAE.p_vqvae.networks import train_transformer_and_vqvae
import torch


vqvae_train_loader_kwargs = {
    "batch_size": 1,
    "augment_flag": True,
    "num_workers": 1
}
training_vqvae_kwargs = {
    "n_epochs": 100,
    "lr": 1e-4,
    "multiple_devices": False,
    "dtype": torch.float32,
    "use_checkpointing": True,
    "commitment_cost": 0.05,
    "num_embeddings": 256,
    "embedding_dim": 32,
    "num_res_layers": 2,
    "num_res_channels": (256, 256),
    "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
    "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    "model_path": "/model_outputs"
}
training_transformer_kwargs = {
    "n_epochs": 50,
    "attn_layers_dim": 96,
    "attn_layers_depth": 12,
    "attn_layers_heads": 8,
    "lr": 5e-4,  # todo: fix "got multiple values for keyword argument 'lr'"
    "model_path": "/model_outputs"
}

if __name__ == "__main__":
    dataset = RawDataSet(
        root="data/ATLAS_2",
        cache_path="data/cache",
        downsample=1,
        normalize=1,
        crop=((8, 9), (12, 13), (0, 9)),
        padding=((1, 2), (0, 0), (1, 2))
    )
    print("cuda available? ", torch.cuda.is_available())
    #train_loader = get_train_loader(dataset, **vqvae_train_loader_kwargs)
    #generator = train_transformer_and_vqvae(train_loader, training_vqvae_kwargs, training_transformer_kwargs,
    #                                        saving_kwargs={"test": "", "downsampling": "1"})