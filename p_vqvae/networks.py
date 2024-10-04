import os
import torch
import time
import numpy as np
import warnings
from tqdm import tqdm
from torch.nn import L1Loss, CrossEntropyLoss
from monai.networks.nets import VQVAE

from monai.utils.ordering import Ordering, OrderingType
from monai.networks.nets import DecoderOnlyTransformer
from monai.inferers import VQVAETransformerInferer


class BaseModel:
    def __init__(self, model, model_path=None, base_filename=""):
        self.trained_flag = False
        self.model = model
        self.model_path = model_path
        self.base_filename = base_filename
        self.full_path = None

    def get_full_path(self, **kwargs):
        filename = self.base_filename
        for key, value in kwargs.items():
            filename += f"_{key}{value}"
        filename += ".pth"
        return os.path.join(self.model_path, filename)

    def save_or_load(self, **kwargs):
        """If weights were saved previously, load them. If not, save them, but only if model was trained."""
        self.full_path = self.get_full_path(**kwargs)
        if os.path.exists(self.full_path) and not self.trained_flag:
            self.load(**kwargs)
        elif os.path.exists(self.full_path) and self.trained_flag:
            response = input('The model you are trying to load has been trained. Overwrite existing weights? '
                             '(y/n)\n')
            if response in ["y", "Yes", "Y"]:
                self.load(**kwargs)
        elif not os.path.exists(self.full_path) and self.trained_flag:
            self.save(**kwargs)
        else:
            warnings.warn(f"Model weights {self.full_path} does not exist. However, the model you are trying to save"
                          f"does not appear to be trained.")
            response = input("Save untrained model weights?\n")
            if response in ["y", "Yes", "Y"]:
                self.save(**kwargs)

    def save(self, **kwargs):
        """
        Saves the model weights with a filename that includes hyperparameters.

        Args:
            model_path: The base directory where the model will be saved.
            **kwargs: Additional keyword arguments representing hyperparameters.
        """
        self.full_path = self.get_full_path(**kwargs)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(), self.full_path)
        print(f"Model saved to {self.full_path}")

    def load(self, **kwargs):
        """
        Loads the model weights based on hyperparameters.

        Args:
            model: The PyTorch model to load weights into.
            model_path: The base directory where the model is saved.
            **kwargs: Keyword arguments representing hyperparameters.
        """
        full_path = self.get_full_path(**kwargs)
        if os.path.exists(full_path):
            state_dict = torch.load(full_path)
            self.model.load_state_dict(state_dict)
            self.trained_flag = True
            print(f"Model loaded from {self.full_path}")
        else:
            print(f"Model file not found: {self.full_path}")


class VQ_VAE(BaseModel):
    def __init__(
        self,
        train_loader,
        val_loader=None,
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(256, 256),
        num_res_channels=(256, 256),
        num_res_layers=2,
        downsample_parameters=((2, 4, 1, 1), (2, 4, 1, 1)),
        upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
        # embeddings:
        num_embeddings=256,
        embedding_dim=32,
        # learning
        lr=1e-4,
        reconstruction_loss=L1Loss(),
        commitment_cost: float = 0.25,  # ensures that encoder output stays close to the selected codebook vector
        n_epochs=100,
        val_interval=5,  # after how many training steps to calculate validation loss
        early_stopping_patience=float('inf'),  # stop training after ... training steps of not improving val loss
        n_example_images=4,  # how many example reconstructions to save in self.final_reconstructions
        dtype=torch.float32,  #
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        multiple_devices=True,
        use_checkpointing=True,
        model_path=None,
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
        self.commitment_cost = commitment_cost
        self.n_epochs = n_epochs
        self.val_interval = 1 if early_stopping_patience else val_interval
        self.early_stopping_patience = early_stopping_patience
        self.n_example_images = n_example_images
        self.device = device
        self.multiple_devices = multiple_devices
        self.use_checkpointing = use_checkpointing
        self.dtype = dtype
        self.model = self._init_model()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.reconstruction_loss = reconstruction_loss
        self.epoch_recon_loss_list = []
        self.epoch_quant_loss_list = []
        self.val_recon_epoch_loss_list = []
        self.intermediary_images = []
        self.final_reconstructions = None
        self.images = None
        super().__init__(self.model, model_path, base_filename="VQVAE")

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
            use_checkpointing=self.use_checkpointing,
            commitment_cost=self.commitment_cost,
        )
        model.to(self.device, dtype=self.dtype)

        # use all GPUS if available
        if self.multiple_devices and torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
        return model

    def train(self):
        val_loss = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_weights = None
        total_start = time.time()
        for epoch in range(self.n_epochs):
            self.model.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch['image'].to(self.device, dtype=self.dtype)

                self.optimizer.zero_grad(set_to_none=True)

                reconstruction, quantization_loss = self.model(images=images)

                if reconstruction.shape != images.shape:
                    print(
                        f"Train Shape mismatch: Reconstruction shape: {reconstruction.shape}, Images shape: "
                        f"{images.shape}"
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

            if (epoch + 1) % self.val_interval == 0 and self.val_loader:
                val_loss = self.validate(epoch)

                if self.early_stopping_patience != float('inf') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_weights = self.model.state_dict()  # Save best weights
                    epochs_without_improvement = 0  # Reset counter if improvement is found
                else:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}.")

                if best_model_weights:
                    print("Loading best model weights so far.")
                    self.model.load_state_dict(best_model_weights)

                break

        self.final_reconstructions = reconstruction[:5].cpu()
        self.images = images[:5].cpu()

        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")

        self.trained_flag = True

        return val_loss

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images = batch['image'].to(self.device, dtype=torch.float32)
                reconstruction, quantization_loss = self.model(images=images)

                if val_step == 1:
                    self.intermediary_images.append(reconstruction[: self.n_example_images, 0])

                recons_loss = self.reconstruction_loss(reconstruction.to(torch.float32), images)

                val_loss += recons_loss.item()

        val_loss /= val_step
        self.val_recon_epoch_loss_list.append(val_loss)
        return val_loss

    def predict(self, img):
        if type(img) == np.ndarray:
            img = torch.from_numpy(img).to(self.device, dtype=self.dtype)
        assert len(img.shape) == 5, f"input shape of image is {len(img.shape)}, should be 5"
        reconstruction, _ = self.model(images=img)
        return reconstruction.cpu()

    def create_synthetic_images(self, num_images=4):
        self.model.eval()

        test_scan = next(iter(self.train_loader))['image'].to(self.device).float()

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

    def get_final_reconstructions_and_images(self):
        return self.final_reconstructions, self.images


class TransformerDecoder_VQVAE(BaseModel):
    def __init__(
        self,
        vqvae_model,
        train_loader,
        val_loader=None,
        attn_layers_dim=96,
        attn_layers_depth=12,
        attn_layers_heads=8,
        lr=5e-4,
        n_epochs=50,
        val_interval=10,
        early_stopping_patience=float('inf'),  # stop training after this many  training steps of not improving val loss
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        multiple_devices=False,
        model_path=None
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vqvae_model = vqvae_model.model
        self.dtype = dtype

        # Transformer hyperparameters
        self.attn_layers_dim = attn_layers_dim
        self.attn_layers_depth = attn_layers_depth
        self.attn_layers_heads = attn_layers_heads
        self.lr = lr
        self.n_epochs = n_epochs

        self.val_interval = val_interval
        self.early_stopping_patience = early_stopping_patience
        self.device = device
        self.multiple_devices = multiple_devices
        self.ce_loss = CrossEntropyLoss()
        self.calculate_intermediate_reconstructions = False
        self.model_path = model_path

        self.ordering = Ordering(
            ordering_type=OrderingType.RASTER_SCAN.value,
            spatial_dims=3,
            dimensions=(1,) + self.vqvae_model.encode_stage_2_inputs(
                next(iter(train_loader))['image'].to(self.device, dtype=self.dtype)).shape[2:],
        )

        self.latent_spatial_dim = self.vqvae_model.encode_stage_2_inputs(
                        next(iter(self.train_loader))['image'].to(self.device, dtype=self.dtype)).shape[2:]

        self.epoch_ce_loss_list = []
        self.val_ce_epoch_loss_list = []
        self.intermediary_images = []

        self.inferer = VQVAETransformerInferer()
        self.transformer_model = self._init_model()
        self.optimizer = torch.optim.Adam(params=self.transformer_model.parameters(), lr=self.lr)

        super().__init__(self.transformer_model, model_path, base_filename="transformer_VQVAE")

    def _init_model(self):
        test_scan = next(iter(self.train_loader))['image'].to(self.device, dtype=self.dtype)
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
        # use all GPUS if available
        if self.multiple_devices and torch.cuda.is_available():
            model = torch.nn.DataParallel(model)
        model = model.to(self.device, dtype=self.dtype)
        return model

    def train(self):
        val_loss = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_weights = None
        self.vqvae_model.eval()
        total_start = time.time()
        for epoch in range(self.n_epochs):
            self.transformer_model.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch['image'].to(self.device).float()

                self.optimizer.zero_grad(set_to_none=True)

                logits, target, _ = self.inferer(
                    images, self.vqvae_model, self.transformer_model, self.ordering, return_latent=True
                )
                logits = logits.transpose(1, 2)

                loss = self.ce_loss(logits, target)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"ce_loss": epoch_loss / (step + 1)})
            self.epoch_ce_loss_list.append(epoch_loss / (step + 1))

            if (epoch + 1) % self.val_interval == 0 and self.val_loader:
                val_loss = self.validate(epoch)

                # early stopping
                if self.early_stopping_patience != float('inf') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_weights = self.transformer_model.state_dict()  # Save best weights
                    epochs_without_improvement = 0  # Reset counter if improvement is found
                else:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}.")

                if best_model_weights:
                    print("Loading best model weights so far.")
                    self.transformer_model.load_state_dict(best_model_weights)

                break

        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")

        self.trained_flag = True

        return val_loss

    def validate(self):
        self.transformer_model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images = batch['image'].to(self.device, dtype=self.dtype)
                logits, quantizations_target, _ = self.inferer(
                    images, self.vqvae_model, self.transformer_model, self.ordering, return_latent=True
                )
                logits = logits.transpose(1, 2)

                loss = self.ce_loss(logits, quantizations_target)

                # Generate a random sample to visualise progress
                if val_step == 1 and self.calculate_intermediate_reconstructions:
                    sample = self.inferer.sample(
                        vqvae_model=self.vqvae_model,
                        transformer_model=self.transformer_model,
                        ordering=self.ordering,
                        latent_spatial_dim=self.latent_spatial_dim,
                        starting_tokens=self.vqvae_model.num_embeddings
                        * torch.ones((1, 1), device=self.device),
                    )
                    self.intermediary_images.append(sample[:, 0].cpu())

                val_loss += loss.item()

        val_loss /= val_step
        self.val_ce_epoch_loss_list.append(val_loss)
        return val_loss

    def predict(self, img):
        if type(img) == np.ndarray:
            img = torch.from_numpy(img)
        prediction = self.inferer.__call__(
            inputs=img,
            vqvae_model=self.vqvae_model,
            transformer_model=self.transformer_model,
            ordering=self.ordering,
        ).cpu()
        return prediction

    def create_synthetic_images(self, num_images=10, temperature=1.0):
        self.vqvae_model.eval()
        self.transformer_model.eval()
        generated_images = []

        with torch.no_grad():
            for i in range(num_images):
                sample = self.inferer.sample(
                    vqvae_model=self.vqvae_model,
                    transformer_model=self.transformer_model,
                    ordering=self.ordering,
                    latent_spatial_dim=self.latent_spatial_dim,
                    starting_tokens=self.vqvae_model.num_embeddings * torch.ones((1, 1), device=self.device),
                    temperature=temperature,
                )
                generated_image = sample[0, 0].cpu().numpy()
                generated_images.append(generated_image)

        return np.stack(generated_images)


def train_transformer_and_vqvae(train_loader, vqvae_training_kwargs: dict, transformer_training_kwargs: dict,
                                saving_kwargs: dict):

    vqvae = VQ_VAE(train_loader, **vqvae_training_kwargs)
    vqvae.train()

    vqvae.save_or_load(**saving_kwargs)

    t_vqvae = TransformerDecoder_VQVAE(vqvae, train_loader, **transformer_training_kwargs)
    t_vqvae.train()
    t_vqvae.save_or_load(**saving_kwargs)

    return t_vqvae
