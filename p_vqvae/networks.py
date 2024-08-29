import os
import torch
import time
import numpy as np
from tqdm import tqdm
from torch.nn import L1Loss, CrossEntropyLoss
from monai.networks.nets import VQVAE

from monai.utils.ordering import Ordering, OrderingType
from monai.networks.nets import DecoderOnlyTransformer
from monai.inferers import VQVAETransformerInferer


class VQ_VAE:
    def __init__(
        self,
        train_loader,
        val_loader,
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(256, 256),
        num_res_channels=(256, 256),
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        num_embeddings=256,
        embedding_dim=32,
        lr=1e-4,
        reconstruction_loss=L1Loss(),
        n_epochs=100,
        val_interval=10,
        n_example_images=4,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_channels = num_res_channels
        self.num_res_layers = num_res_layers
        self.downsample_parameters = downsample_parameters
        self.upsample_parameters = upsample_parameters
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.n_epochs = n_epochs
        self.val_interval = val_interval
        self.n_example_images = n_example_images
        self.device = device
        self.model = self._init_model()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.reconstruction_loss = reconstruction_loss
        self.epoch_recon_loss_list = []
        self.epoch_quant_loss_list = []
        self.val_recon_epoch_loss_list = []
        self.intermediary_images = []

    def _init_model(self):
        model = VQVAE(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.channels,
            num_res_channels=self.num_res_channels,
            num_res_layers=self.num_res_layers,
            downsample_parameters=self.downsample_parameters,
            upsample_parameters=self.upsample_parameters,
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
        )
        model.to(self.device, dtype=torch.float32)
        return model

    def train(self):
        total_start = time.time()
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch.to(self.device, dtype=torch.float32)

                self.optimizer.zero_grad(set_to_none=True)

                reconstruction, quantization_loss = self.model(images=images)

                if reconstruction.shape != images.shape:
                    print(
                        f"Train Shape mismatch: Reconstruction shape: {reconstruction.shape}, Images shape: {images.shape}"
                    )

                recons_loss = self.reconstruction_loss(reconstruction.to(torch.float16), images)
                loss = recons_loss + quantization_loss

                loss.backward()
                self.optimizer.step()

                epoch_loss += recons_loss.item()

                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "quantization_loss": quantization_loss.item() / (step + 1),
                    }
                )
            self.epoch_recon_loss_list.append(epoch_loss / (step + 1))
            self.epoch_quant_loss_list.append(quantization_loss.item() / (step + 1))

            if (epoch + 1) % self.val_interval == 0:
                self.validate(epoch)

        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images = batch.to(self.device, dtype=torch.float32)
                reconstruction, quantization_loss = self.model(images=images)

                if val_step == 1:
                    self.intermediary_images.append(reconstruction[: self.n_example_images, 0])

                recons_loss = self.reconstruction_loss(reconstruction.to(torch.float32), images)

                val_loss += recons_loss.item()

        val_loss /= val_step
        self.val_recon_epoch_loss_list.append(val_loss)

    def create_synthetic_images(self, num_images=4):
        self.model.eval()

        test_scan = next(iter(self.train_loader)).to(self.device).float()

        latent_shape = self.model.encode(test_scan).shape
        spatial_shape = latent_shape[2:]

        random_indices = torch.randint(
            low=0,
            high=self.model.num_embeddings,
            size=(num_images, *spatial_shape),
        ).to(self.device)

        with torch.no_grad():
            synthetic_images = self.model.decode_samples(random_indices)

        return synthetic_images.cpu().numpy()

    def save(self, model_path, **kwargs):
        """
        Saves the model weights with a filename that includes hyperparameters.

        Args:
            model_path: The base directory where the model will be saved.
            **kwargs: Additional keyword arguments representing hyperparameters.
        """
        filename = "VQVAE"
        for key, value in kwargs.items():
            filename += f"_{key}{value}"
        filename += ".pth"
        full_path = os.path.join(model_path, filename)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, model_path, **kwargs):
        """
        Loads the model weights based on hyperparameters.

        Args:
            model: The PyTorch model to load weights into.
            model_path: The base directory where the model is saved.
            **kwargs: Keyword arguments representing hyperparameters.
        """
        filename = "VQVAE"
        for key, value in kwargs.items():
            filename += f"_{key}{value}"
        filename += ".pth"
        full_path = os.path.join(model_path, filename)
        if os.path.exists(full_path):
            state_dict = torch.load(full_path)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {full_path}")
        else:
            print(f"Model file not found: {full_path}")


class TransformerDecoder_VQVAE:
    def __init__(
        self,
        train_loader,
        val_loader,
        vqvae_model,
        attn_layers_dim=96,
        attn_layers_depth=12,
        attn_layers_heads=8,
        lr=5e-4,
        n_epochs=50,
        val_interval=10,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vqvae_model = vqvae_model
        self.dtype = dtype

        # Transformer hyperparameters
        self.attn_layers_dim = attn_layers_dim
        self.attn_layers_depth = attn_layers_depth
        self.attn_layers_heads = attn_layers_heads
        self.lr = lr
        self.n_epochs = n_epochs

        self.val_interval = val_interval
        self.device = device
        self.optimizer = torch.optim.Adam(params=self.vqvae_model.parameters(), lr=self.lr)
        self.ce_loss = CrossEntropyLoss()

        self.ordering = Ordering(
            ordering_type=OrderingType.RASTER_SCAN.value,
            spatial_dims=3,
            dimensions=(1,) + self.vqvae_model.encode_stage_2_inputs(next(iter(train_loader))).shape[2:],
        )

        self.epoch_ce_loss_list = []
        self.val_ce_epoch_loss_list = []
        self.intermediary_images = []

        self.inferer = VQVAETransformerInferer()
        self.model = self._init_model()

    def _init_model(self):
        test_scan = next(iter(self.train_loader)).to(self.device, dtype=self.dtype)
        spatial_shape = self.vqvae_model.encode_stage_2_inputs(test_scan).shape[2:]

        # define maximum sequence length
        max_seq_len = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]

        # Beginning of sentence token + 1
        num_tokens = self.vqvae_model.num_embeddings + 1

        model = DecoderOnlyTransformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers_dim=self.attn_layers_dim,
            attn_layers_depth=self.attn_layers_depth,
            attn_layers_heads=self.attn_layers_heads,
        )
        model = model.to(self.device, dtype=self.dtype)
        return model

    def train(self):
        self.vqvae_model.eval()
        total_start = time.time()
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch.to(self.device).float()

                self.optimizer.zero_grad(set_to_none=True)

                logits, target, _ = self.inferer(
                    images, self.vqvae_model, self.model, self.ordering, return_latent=True
                )
                logits = logits.transpose(1, 2)

                loss = self.ce_loss(logits, target)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"ce_loss": epoch_loss / (step + 1)})
            self.epoch_ce_loss_list.append(epoch_loss / (step + 1))

            if (epoch + 1) % self.val_interval == 0:
                self.validate()

        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images = batch.to(self.device).float()
                logits, quantizations_target, _ = self.inferer(
                    images, self.vqvae_model, self.model, self.ordering, return_latent=True
                )
                logits = logits.transpose(1, 2)

                loss = self.ce_loss(logits, quantizations_target)

                # Generate a random sample to visualise progress
                if val_step == 1:
                    spatial_shape = self.vqvae_model.encode_stage_2_inputs(next(iter(self.train_loader))).shape[2:]
                    sample = self.inferer.sample(
                        vqvae_model=self.vqvae_model,
                        transformer_model=self.model,
                        ordering=self.ordering,
                        latent_spatial_dim=spatial_shape,
                        starting_tokens=self.vqvae_model.num_embeddings
                        * torch.ones((1, 1), device=self.device),
                    )
                    self.intermediary_images.append(sample[:, 0])

                val_loss += loss.item()

        val_loss /= val_step
        self.val_ce_epoch_loss_list.append(val_loss)

    def create_synthetic_images(self, num_images=10):
        self.vqvae_model.eval()
        self.model.eval()
        generated_images = []

        with torch.no_grad():
            for i in range(num_images):
                sample = self.inferer.sample(
                    vqvae_model=self.vqvae_model,
                    transformer_model=self.model,
                    ordering=self.ordering,
                    latent_spatial_dim=self.vqvae_model.encode_stage_2_inputs(next(iter(self.train_loader))).shape[2:],
                    starting_tokens=self.vqvae_model.num_embeddings * torch.ones((1, 1), device=self.device),
                )
                generated_image = sample[0, 0].cpu().numpy()
                generated_images.append(generated_image)

        return np.stack(generated_images)

    def save(self, model_path, **kwargs):
        """
        Saves the model weights with a filename that includes hyperparameters.

        Args:
            model_path: The base directory where the model will be saved.
            **kwargs: Additional keyword arguments representing hyperparameters.
        """
        filename = "VQVAE"
        for key, value in kwargs.items():
            filename += f"_{key}{value}"
        filename += ".pth"
        full_path = os.path.join(model_path, filename)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved to {full_path}")

    def load(self, model_path, **kwargs):
        """
        Loads the model weights based on hyperparameters.

        Args:
            model: The PyTorch model to load weights into.
            model_path: The base directory where the model is saved.
            **kwargs: Keyword arguments representing hyperparameters.
        """
        filename = "VQVAE"
        for key, value in kwargs.items():
            filename += f"_{key}{value}"
        filename += ".pth"
        full_path = os.path.join(model_path, filename)
        if os.path.exists(full_path):
            state_dict = torch.load(full_path)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {full_path}")
        else:
            print(f"Model file not found: {full_path}")


