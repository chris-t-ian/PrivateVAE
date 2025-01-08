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
    def __init__(self, model, model_path=None, base_filename="", device=None, device_ids=None, seed=None):
        self.trained_flag = False
        self.model_path = model_path
        self.base_filename = base_filename
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if seed and "cuda" in str(self.device):
            torch.cuda.manual_seed(seed)  # set fixed seed
        elif seed and self.device == "cpu":
            torch.manual_seed(seed)  # set fixed seed
        elif seed is None and "cuda" in str(self.device):
            torch.cuda.seed_all()  # set fresh random seed
        elif seed is None and self.device == "cpu":
            torch.seed()  # set fresh random seed

        if device_ids is not None:
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.model = model.to(self.device)

    def get_full_path(self, **kwargs):
        filename = self.base_filename
        for key, value in kwargs.items():
            filename += f"_{key}{value}"
        filename += ".pth"
        return os.path.join(self.model_path, filename)

    def model_exists(self, **kwargs):
        return os.path.exists(self.get_full_path(**kwargs))

    def save_or_load(self, **kwargs):
        """If weights were saved previously, load them. If not, save them, but only if model was trained."""
        if self.model_exists(**kwargs) and not self.trained_flag:
            self.load(**kwargs)
        elif self.model_exists(**kwargs) and self.trained_flag:
            # response = input('The model you are trying to load has been trained. Overwrite existing weights? '
            #                 '(y/n)\n')
            # if response in ["y", "Yes", "Y"]:
            #   self.load(**kwargs)
            pass
        elif not self.model_exists(**kwargs) and self.trained_flag:
            self.save(**kwargs)
        else:
            warnings.warn(f"Model weights {self.get_full_path(**kwargs)} does not exist. However, the model you are"
                          f"trying to save does not appear to be trained.")
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
        full_path = self.get_full_path(**kwargs)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        torch.save(self.model.state_dict(), full_path)
        print(f"Model saved to {full_path}")

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

            if "module" in state_dict.keys():
                from collections import OrderedDict
                new_state_dict = OrderedDict()

                for key, value in state_dict.items():
                    new_key = key.replace("module.", "")  # Remove 'module.' from the key
                    new_state_dict[new_key] = value
                    print("removing")

                state_dict = new_state_dict

            self.model.load_state_dict(state_dict)
            self.trained_flag = True
            print(f"Model loaded from {full_path}")
        else:
            print(f"Model file not found: {full_path}")


class MembershipClassifierModule(torch.nn.Module):
    def __init__(self, input_dims, num_classes, hidden_channels):
        super(MembershipClassifierModule, self).__init__()

        self.input_dims = input_dims
        print("input dims: ", self.input_dims)
        self.hidden_channels = hidden_channels

        self.conv1 = torch.nn.Conv3d(1, hidden_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv3d(hidden_channels[0], hidden_channels[1], kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv3d(hidden_channels[1], hidden_channels[2], kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool3d(kernel_size=2, stride=2)

        self.fc1 = None
        self.fc2 = torch.nn.Linear(128, num_classes)

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # First 3D Conv layer
        x = self.pool(torch.relu(self.conv2(x)))  # Second 3D Conv layer
        x = self.pool(torch.relu(self.conv3(x)))  # Third 3D Conv layer

        flattened_dim = x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]  # Correct flattened size

        # Initialize fc1 if it hasn't been initialized yet
        if self.fc1 is None:
            self.fc1 = torch.nn.Linear(flattened_dim, 128).to(x.device)

        x = x.view(-1, flattened_dim)  # Flatten 3D feature maps to 1D
        x = torch.relu(self.fc1(x))  # Fully connected layer
        x = self.dropout(x)  # Dropout for regularization
        x = self.fc2(x)  # Output layer

        return x


class MembershipClassifier(BaseModel):
    def __init__(
            self,
            train_loader,
            val_loader=None,
            channels=(16, 32, 64),
            epochs=100,
            val_interval=10,
            model_path=None,
            device=None,
            seed=None
    ):
        self.train_loader = train_loader
        self.shape = next(iter(self.train_loader))['image'].shape
        self.val_loader = val_loader
        self.channels = channels
        self.n_epochs = epochs
        self.val_interval = val_interval
        self.early_stopping_patience = float('inf')

        model = self._init_model()
        super().__init__(model, model_path, "MembershipCNN", device, seed)

        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def _init_model(self):
        return MembershipClassifierModule(self.shape[2:], num_classes=2, hidden_channels=self.channels)

    def train(self):
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_weights = None
        for epoch in range(self.n_epochs):

            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                # TODO: shuffle labels
                print("images: ", images.shape)
                print("labels: ", labels)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                progress_bar.set_postfix({"loss": loss.item()})

            if (epoch + 1) % self.val_interval == 0 and self.val_loader:
                val_loss = self.validate()

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

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.loss(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        accuracy = correct / len(self.val_loader.dataset)
        return running_loss / len(self.val_loader.dataset), accuracy

    def logits(self, _input):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(_input)

        return logits


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
        use_checkpointing=True,
        model_path=None,
        seed=None,
        device=None,
        device_ids: list = None,
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
        self.use_checkpointing = use_checkpointing
        self.dtype = dtype
        self.reconstruction_loss = reconstruction_loss
        self.epoch_recon_loss_list = []
        self.epoch_quant_loss_list = []
        self.val_recon_epoch_loss_list = []
        self.intermediary_images = []
        self.final_reconstructions = None
        self.images = None
        model = self._init_model()
        super().__init__(model, model_path, "VQVAE", device, device_ids, seed)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

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
                images = batch.to(self.device, dtype=self.dtype)
                self.optimizer.zero_grad(set_to_none=True)

                reconstruction, quantization_loss = self.model(images=images)

                if reconstruction.shape != images.shape:
                    print(
                        f"Train Shape mismatch: Reconstruction shape: {reconstruction.shape}, Images shape: "
                        f"{images.shape}"
                    )

                recons_loss = self.reconstruction_loss(reconstruction.to(torch.float16), images)
                loss = recons_loss + quantization_loss

                if "DataParallel" in str(type(self.model)):
                    loss = loss.mean()

                loss.backward()
                self.optimizer.step()

                epoch_loss += recons_loss.item()

                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "quantization_loss": quantization_loss / (step + 1),
                    }
                )
            self.epoch_recon_loss_list.append(epoch_loss / (step + 1))
            # self.epoch_quant_loss_list.append(quantization_loss[0].item() / (step + 1))

            if epoch % self.val_interval == 0 and self.val_loader:
                val_loss = self.validate()

                if self.early_stopping_patience != float('inf') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_weights = self.model.state_dict()  # Save best weights
                    epochs_without_improvement = 0  # Reset counter if improvement is found
                else:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                best_epoch = epoch - (self.val_interval * epochs_without_improvement)
                print(f"Early stopping at epoch {epoch}. Loading weights of epoch {best_epoch}. ")

                with open(os.path.join(self.model_path, "VQVAE_early_stopping.txt") , 'w') as f:
                    f.write(f"Early stopping at epoch {best_epoch}. Best val loss at epoch {best_epoch}. \n"
                            f"Path: {self.model_path}")

                if best_model_weights:
                    print("Loading best model weights so far.")
                    self.model.load_state_dict(best_model_weights)

                break

        self.final_reconstructions = reconstruction[:5].cpu()
        self.images = images[:5].cpu()

        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")

        self.trained_flag = True

        return best_val_loss

    def validate(self):
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
        return val_loss

    def predict(self, img):
        if type(img) == np.ndarray:
            img = torch.from_numpy(img).to(self.device, dtype=self.dtype)
        assert len(img.shape) == 5, f"input shape of image is {len(img.shape)}, should be 5"
        reconstruction, _ = self.model(images=img)
        return reconstruction.cpu()

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

    def get_final_reconstructions_and_images(self):
        return self.final_reconstructions, self.images


class TransformerDecoder_VQVAE(BaseModel):
    def __init__(
        self,
        vqvae,
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
        model_path=None,
        seed=None,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        vqvae_device=None,
        device_ids: list = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dtype = dtype
        self.device = device
        self.vqvae_device = device if vqvae_device is None else vqvae_device

        if "parallel" in str(type(vqvae.model)):
            self.vqvae_model = vqvae.model.module.to(device=vqvae_device)
        else:
            self.vqvae_model = vqvae.model.to(device=vqvae_device)

        # Transformer hyperparameters
        self.attn_layers_dim = attn_layers_dim
        self.attn_layers_depth = attn_layers_depth
        self.attn_layers_heads = attn_layers_heads
        self.lr = lr
        self.n_epochs = n_epochs

        self.val_interval = val_interval
        self.early_stopping_patience = early_stopping_patience
        self.ce_loss = CrossEntropyLoss()
        self.calculate_intermediate_reconstructions = False
        self.model_path = model_path

        self.epoch_ce_loss_list = []
        self.val_ce_epoch_loss_list = []
        self.intermediary_images = []

        self.inferer = VQVAETransformerInferer()
        transformer_model = self._init_model()
        self.seed = seed
        super().__init__(transformer_model, model_path, "transformer_VQVAE", device, device_ids, seed)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.latent_spatial_dim = self.vqvae_model.encode_stage_2_inputs(
            next(iter(self.train_loader)).to(device=self.vqvae_device, dtype=self.dtype)).shape[2:]
        self.ordering = Ordering(
            ordering_type=OrderingType.RASTER_SCAN.value,
            spatial_dims=3,
            dimensions=(1,) + self.vqvae_model.encode_stage_2_inputs(
                next(iter(train_loader)).to(device=self.vqvae_device, dtype=self.dtype)).shape[2:],
        )

    def _init_model(self):
        test_scan = next(iter(self.train_loader)).to(self.vqvae_device, dtype=self.dtype)
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

        return model

    def train(self):
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_weights = None
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

            if epoch % self.val_interval == 0 and self.val_loader:
                val_loss = self.validate()

                # early stopping
                if self.early_stopping_patience != float('inf') and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_weights = self.model.state_dict()  # Save best weights
                    epochs_without_improvement = 0  # Reset counter if improvement is found
                else:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                best_epoch = epoch - (self.val_interval * epochs_without_improvement)
                print(f"Early stopping at epoch {epoch}. Loading weights of epoch {best_epoch}.")

                with open(os.path.join(self.model_path, "VQVAE_early_stopping.txt") , 'w') as f:
                    f.write(f"Early stopping at epoch {best_epoch}. Best val loss at epoch {best_epoch}. \n"
                            f"Path: {self.model_path}")

                if best_model_weights:
                    print("Loading best model weights so far.")
                    self.model.load_state_dict(best_model_weights)

                break

        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")

        self.trained_flag = True

        return best_val_loss

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(self.val_loader, start=1):
                images = batch.to(self.device, dtype=self.dtype)
                logits, quantizations_target, _ = self.inferer(
                    images, self.vqvae_model, self.model, self.ordering, return_latent=True
                )
                logits = logits.transpose(1, 2)

                loss = self.ce_loss(logits, quantizations_target)

                # Generate a random sample to visualise progress
                if val_step == 1 and self.calculate_intermediate_reconstructions:
                    sample = self.inferer.sample(
                        vqvae_model=self.vqvae_model,
                        transformer_model=self.model,
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
            transformer_model=self.model,
            ordering=self.ordering,
        ).cpu()
        return prediction

    def create_synthetic_images(self, num_images=10, temperature=1.0):
        self.vqvae_model.eval()
        self.model.eval()
        generated_images = []

        seed_used = torch.initial_seed() if self.device == "cpu" else torch.cuda.initial_seed()
        if seed_used != self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        with torch.no_grad():
            for i in range(num_images):
                sample = self.inferer.sample(
                    vqvae_model=self.vqvae_model,
                    transformer_model=self.model,
                    ordering=self.ordering,
                    latent_spatial_dim=self.latent_spatial_dim,
                    starting_tokens=self.vqvae_model.num_embeddings * torch.ones((1, 1), device=self.device),
                    temperature=temperature,
                )
                generated_image = sample[0, 0].cpu().numpy()
                generated_images.append(generated_image)

        generated_images = np.stack(generated_images)
        return np.expand_dims(generated_images, axis=1)


class ResidualBlock(torch.nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs.
    Source: https://github.com/bayesiains/nflows/blob/master/nflows/nn/nets/resnet.py"""

    def __init__(
            self,
            features,
            context_features,
            activation=torch.nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
            zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = torch.nn.Linear(context_features, features)
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = torch.nn.Dropout(p=dropout_probability)
        if zero_initialization:
            torch.nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            torch.nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = torch.nn.functional.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps


class ResidualNet(torch.nn.Module):
    """A general-purpose residual network. Works only with 1-dim inputs.
    Source: https://github.com/bayesiains/nflows/blob/master/nflows/nn/nets/resnet.py"""

    def __init__(
            self,
            in_features,
            out_features,
            hidden_features,
            context_features=None,
            num_blocks=2,
            activation=torch.nn.functional.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = torch.nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = torch.nn.Linear(in_features, hidden_features)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = torch.nn.Linear(hidden_features, out_features)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)
        return outputs


def train_transformer_and_vqvae(train_loader, vqvae_kwargs: dict, transformer_kwargs: dict,
                                saving_kwargs: dict, val_loader=None, seed=None, return_vqvae=False):

    vqvae = VQ_VAE(train_loader, val_loader, seed=seed, **vqvae_kwargs)
    if vqvae.model_exists(**saving_kwargs):
        print("vqvae model already saved")
        vqvae.load(**saving_kwargs)
    else:
        vqvae.train()
        vqvae.save(**saving_kwargs)

    t_vqvae = TransformerDecoder_VQVAE(vqvae, train_loader, val_loader, seed=seed, **transformer_kwargs)
    if t_vqvae.model_exists(**saving_kwargs):
        t_vqvae.load(**saving_kwargs)
    else:
        t_vqvae.train()
        t_vqvae.save(**saving_kwargs)

    if return_vqvae:
        return t_vqvae, vqvae
    else:
        return t_vqvae
