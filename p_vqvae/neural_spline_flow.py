import math
import time
import nflows
from tqdm import tqdm
from nflows.distributions import StandardNormal
from nflows.transforms.splines import rational_quadratic_spline
from nflows.utils import create_mid_split_binary_mask
from nflows.flows import Flow
from nflows.transforms import CompositeTransform, MultiscaleCompositeTransform, Transform
from nflows.transforms import ActNorm as _ActNorm
from nflows.transforms import PiecewiseRationalQuadraticCouplingTransform as \
    _PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
import torch

from p_vqvae.dataloader import DataSet, get_train_val_loader, load_batches
from p_vqvae.networks import BaseModel


data_kwargs = {
    "root": "/home/chrsch/P_VQVAE/data/ATLAS_2",
    "cache_path": '/home/chrsch/P_VQVAE/data/cache/',
    "downsample": 4,
    "normalize": 1,
    "crop": ((8, 9), (12, 13), (0, 9)),
    "padding": ((1, 2), (0, 0), (1, 2)),
}
train_loader_kwargs = {
    "batch_size": 1,
    "augment_flag": True,
    "num_workers": 2
}
model_params = {
    "model_path": "/home/chrsch/P_VQVAE/model_outputs/lira",
    "device": "cuda",
    "steps_per_level": 10,
    "levels": 2,  # increase for non-downsampled dataset
    "multi_scale": True,
    "actnorm": False,
}
_spline_params = {
        'num_bins': 4,
        'tail_bound': 1.,
        'min_bin_width': 1e-3,
        'min_bin_height': 1e-3,
        'min_derivative': 1e-3,
        'apply_unconditional_transform': False
}
optimization = {
    "batch_size": 8,
    "learning_rate": 1e-5,
    "cosine_annealing": False,
    "eta_min": 0.,
    "warmup_fraction": 0.,
    "num_steps": 10000,
    "temperatures": [0.5, 0.75, 1.],
}
coupling_transform = {
    "coupling_layer_type": 'rational_quadratic_spline',
    "hidden_channels": 64,
    "use_resnet": False,
    "num_res_blocks": 5,  # If using resnet
    "resnet_batchnorm": True,
    "dropout_prob": 0.,
}
intervals_ = {
        'save': 1000,
        'sample': 1000,
        'eval': 50,
        'reconstruct': 1000,
        'log': 50
    }
nsf_params = {**model_params, **optimization, **coupling_transform, "spline_parameters": _spline_params, "intervals":
              intervals_}


class Conv3dSameSize(torch.nn.Conv3d):
    """Makes sure that the output has the same shape as the input. Adaptation of Conv2dSameSize of
    nsf.experiments.autils for 3d data."""
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        # print("padding: ", same_padding)  # debugging print
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding)


class ConvNet(torch.nn.Module):
    """Encoder to reduce dimensions of MRI data."""
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = torch.nn.Sequential(
            Conv3dSameSize(in_channels, hidden_channels, kernel_size=3),
            torch.nn.ReLU(),
            Conv3dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            torch.nn.ReLU(),
            Conv3dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, x, context=None):
        return self.net.forward(x)


def create_transform_step(num_channels, hidden_channels, actnorm, spline_params, coupling_layer_type,
                          use_resnet, dropout_prob, steps_per_level=None):
    if use_resnet:
        raise NotImplementedError
    else:
        if dropout_prob != 0.:
            raise ValueError()

        def create_convnet(in_channels, out_channels):
            return ConvNet(in_channels, out_channels, hidden_channels=hidden_channels)

    mask = nflows.utils.create_mid_split_binary_mask(num_channels)
    # print("mask.shape: ", mask.shape)  # debugging print
    # print("Number of ones in mask:", mask.sum())

    if coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=spline_params['num_bins'],
            apply_unconditional_transform=spline_params['apply_unconditional_transform'],
            min_bin_width=spline_params['min_bin_width'],
            min_bin_height=spline_params['min_bin_height'],
            min_derivative=spline_params['min_derivative']
        )
    else:
        raise RuntimeError('Unknown coupling_layer_type')

    step_transforms = []

    if actnorm:
        step_transforms.append(ActNorm(num_channels))

    step_transforms.extend([
        OneByOneConvolution(num_channels),
        coupling_layer
    ])

    return CompositeTransform(step_transforms)


class SqueezeTransform(Transform):
    """A transformation defined for 3D image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.

    Implementation adapted from https://github.com/pclucas14/pytorch-glow,
    https://github.com/chaiyujin/glow-pytorch and https://github.com/bayesiains/nsf


    References:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    > C. Durkan et al., Neural Spline Flows, NeurIPS 2019.
    """
    def __init__(self, factor=2):
        super(SqueezeTransform, self).__init__()

        if not type(factor) == int or factor <= 1:
            raise ValueError('Factor must be an integer > 1.')

        self.factor = factor

    def get_output_shape(self, c, h, w, d):
        return (c * self.factor * self.factor * self.factor,
                h // self.factor,
                w // self.factor,
                d // self.factor)

    def forward(self, inputs, context=None):
        if inputs.dim() != 5:
            raise ValueError('Expecting inputs with 5 dimensions')

        batch_size, c, h, w, d = inputs.size()

        if h % self.factor != 0 or w % self.factor != 0 or d % self.factor != 0:
            raise ValueError('Input image size not compatible with the factor.')

        inputs = inputs.view(batch_size, c, h // self.factor, self.factor, w // self.factor,
                             self.factor, d // self.factor, self.factor)
        inputs = inputs.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        inputs = inputs.view(batch_size, c * self.factor * self.factor * self.factor, h // self.factor,
                             w // self.factor, d // self.factor)
        # print("inputs shape after squeezing: ", inputs.shape)
        return inputs, torch.zeros(batch_size)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 5:
            raise ValueError('Expecting inputs with 5 dimensions')

        batch_size, c, h, w, d = inputs.size()

        if c % (self.factor ** 3) != 0:
            raise ValueError(f'Invalid number of channel dimensions: {c}. '
                             f'It must be divisible by {self.factor ** 3}.')

        inputs = inputs.view(batch_size, c // self.factor ** 3, self.factor, self.factor, self.factor, h, w, d)
        inputs = inputs.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        inputs = inputs.view(batch_size, c // self.factor ** 3, h * self.factor, w * self.factor, d * self.factor)
        print("inputs shape after inverse squeezing: ", inputs.shape)
        return inputs, torch.zeros(batch_size)


class ReshapeTransform(nflows.transforms.Transform):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.input_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.input_shape))
        return inputs.reshape(-1, *self.output_shape), torch.zeros(inputs.shape[0])

    def inverse(self, inputs, context=None):
        if tuple(inputs.shape[1:]) != self.output_shape:
            raise RuntimeError('Unexpected inputs shape ({}, but expecting {})'
                               .format(tuple(inputs.shape[1:]), self.output_shape))
        return inputs.reshape(-1, *self.input_shape), torch.zeros(inputs.shape[0])


class ActNorm(_ActNorm):
    def _broadcastable_scale_shift(self, inputs):
        if inputs.dim() == 5:
            return self.scale.view(1, -1, 1, 1, 1), self.shift.view(1, -1, 1, 1, 1)
        elif inputs.dim() == 4:
            return self.scale.view(1, -1, 1, 1), self.shift.view(1, -1, 1, 1)
        else:
            return self.scale.view(1, -1), self.shift.view(1, -1)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4, 5]:
            raise ValueError("Expecting inputs to be a 2D, 4D or a 5D tensor.")

        if self.training and not self.initialized:
            self._initialize(inputs)

        if torch.all(inputs.eq(0)):
            print("Encountered inputs tensor containing only 0.")

        scale, shift = self._broadcastable_scale_shift(inputs)

        if torch.all(scale.eq(0)):
            print("Encountered scale tensor containing only 0.")

        scale = torch.nan_to_num(scale, nan=0.0, posinf=1e6, neginf=-1e6)
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        shift = torch.nan_to_num(shift, nan=0.0, posinf=1e6, neginf=-1e6)

        outputs = scale * inputs + shift

        if torch.isnan(self.log_scale).any():
            self.log_scale.data = torch.nan_to_num(self.log_scale.data, nan=0.0, posinf=1e6, neginf=-1e6)

        if inputs.dim() == 5:
            batch_size, _, h, w, d = inputs.shape
            logabsdet = h * w * d * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        elif inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        else:
            batch_size, _ = inputs.shape
            logabsdet = torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        if torch.isnan(self.log_scale).any():
            print(f"self.logscale in Actnorm contains NaNs.")
        if torch.isnan(torch.sum(self.log_scale)).any():
            print("sum of logscale in Actnorm contains NaNs")
        if torch.isnan(outputs).any():
            print("ouptuts in actnorm contains nans")

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4, 5]:
            raise ValueError("Expecting inputs to be a 2D, 4D or a 5D tensor.")

        scale, shift = self._broadcastable_scale_shift(inputs)
        outputs = (inputs - shift) / scale

        if inputs.dim() == 5:
            batch_size, _, h, w, d = inputs.shape
            logabsdet = -h * w * d * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        elif inputs.dim() == 4:
            batch_size, _, h, w = inputs.shape
            logabsdet = -h * w * torch.sum(self.log_scale) * outputs.new_ones(batch_size)
        else:
            batch_size, _ = inputs.shape
            logabsdet = -torch.sum(self.log_scale) * outputs.new_ones(batch_size)

        return outputs, logabsdet

    def _initialize(self, inputs, epsilon=1e-6):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        if inputs.dim() == 4:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 1).reshape(-1, num_channels)
        elif inputs.dim() == 5:
            num_channels = inputs.shape[1]
            inputs = inputs.permute(0, 2, 3, 4, 1).reshape(-1, num_channels)

        with torch.no_grad():
            std = inputs.std(dim=0)
            std = torch.clamp(std, min=epsilon)  # added small epsilon so that
            mu = (inputs / std).mean(dim=0)
            print("std: ", std)
            print("mu: ", mu)

            self.log_scale.data = -torch.log(std)

            if torch.isinf(self.log_scale.data).any():
                print(f"log scale data has {torch.sum(torch.isinf(self.log_scale))}"
                      f"inf values. Total values: {torch.numel(self.log_scale)}")
            if torch.isnan(self.log_scale.data).any():
                print(f"log scale data has {torch.sum(torch.isnan(self.log_scale))}"
                      f"inf values. Total values: {torch.numel(self.log_scale)}")
            if torch.isnan(self.log_scale).any():
                print(f"log scale has {torch.sum(torch.isnan(self.log_scale))}"
                      f"inf values. Total values: {torch.numel(self.log_scale)}")
            print(self.log_scale.data)
            self.shift.data = -mu
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class OneByOneConvolution(LULinear):
    """An invertible 1x1 convolution with a fixed permutation, as introduced in the Glow paper.

        Reference:
        > D. Kingma et. al., Glow: Generative flow with invertible 1x1 convolutions, NeurIPS 2018.
        """

    def __init__(self, num_channels, using_cache=False, identity_init=True):
        super().__init__(num_channels, using_cache, identity_init)
        self.permutation = RandomPermutation(num_channels, dim=1)

    def _lu_forward_inverse(self, inputs, inverse=False):
        from nflows.utils import torchutils
        b, c, h, w, d = inputs.shape
        inputs = inputs.permute(0, 2, 3, 4, 1).reshape(b * h * w * d, c)

        if inverse:
            outputs, logabsdet = super().inverse(inputs)
        else:
            outputs, logabsdet = super().forward(inputs)

        outputs = outputs.reshape(b, h, w, d, c).permute(0, 4, 1, 2, 3)
        logabsdet = logabsdet.reshape(b, h, w, d)

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        if inputs.dim() != 5:
            raise ValueError("Inputs must be a 5D tensor.")

        inputs, _ = self.permutation(inputs)

        return self._lu_forward_inverse(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 5:
            raise ValueError("Inputs must be a 5D tensor.")

        outputs, logabsdet = self._lu_forward_inverse(inputs, inverse=True)

        outputs, _ = self.permutation.inverse(outputs)

        return outputs, logabsdet


class PiecewiseRationalQuadraticCouplingTransform(_PiecewiseRationalQuadraticCouplingTransform):
    def _coupling_transform(self, inputs, transform_params, inverse=False):
        from nflows.utils import torchutils
        if inputs.dim() == 5:
            b, c, h, w, d = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w, d).permute(0, 1, 3, 4, 5, 2)
        elif inputs.dim() == 4:
            b, c, h, w = inputs.shape
            # For images, reshape transform_params from Bx(C*?)xHxW to BxCxHxWx?
            transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        elif inputs.dim() == 2:
            b, d = inputs.shape
            # For 2D data, reshape transform_params from Bx(D*?) to BxDx?
            transform_params = transform_params.reshape(b, d, -1)

        outputs, logabsdet = self._piecewise_cdf(inputs, transform_params, inverse)

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [2, 4, 5]:
            raise ValueError('Inputs must be a 2D, 4D or a 5D tensor.')
        if torch.isnan(inputs).any():
            print(f"The tensor in coupling_transform forward contains {torch.sum(torch.isnan(inputs))} NaN values. It"
                  f"contains {torch.numel(inputs)} values in total.")

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
               self.features, inputs.shape[1]))
            pass

        identity_split = inputs[:, self.identity_features, ...]
        transform_split = inputs[:, self.transform_features, ...]
        assert transform_split.numel() > 0, "transform_split is empty"

        transform_params = self.transform_net(identity_split, context)

        transform_split, logabsdet = self._coupling_transform_forward(
            inputs=transform_split,
            transform_params=transform_params
        )

        if self.unconditional_transform is not None:
            identity_split, logabsdet_identity =\
                self.unconditional_transform(identity_split, context)
            logabsdet += logabsdet_identity

        outputs = torch.empty_like(inputs)
        identity_features = self.identity_features.to(device=outputs.device)
        transform_features = self.transform_features.to(device=outputs.device)
        outputs[:, identity_features, ...] = identity_split
        outputs[:, transform_features, ...] = transform_split

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [2, 4, 5]:
            raise ValueError('Inputs must be a 2D, 4D or a 5D tensor.')

        if inputs.shape[1] != self.features:
            raise ValueError('Expected features = {}, got {}.'.format(
                self.features, inputs.shape[1]))

        identity_features = self.identity_features.to(device=inputs.device)
        transform_features = self.transform_features.to(device=inputs.device)
        identity_split = inputs[:, identity_features, ...]
        transform_split = inputs[:, transform_features, ...]

        logabsdet = 0.0
        if self.unconditional_transform is not None:
            identity_split, logabsdet = self.unconditional_transform.inverse(identity_split,
                                                                             context)

        transform_params = self.transform_net(identity_split, context)
        transform_split, logabsdet_split = self._coupling_transform_inverse(
            inputs=transform_split,
            transform_params=transform_params
        )
        logabsdet += logabsdet_split

        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_features] = identity_split
        outputs[:, self.transform_features] = transform_split

        return outputs, logabsdet


def nats_to_bits_per_dim(nats, _c, _h, _w, _d):
    return nats / (math.log(2) * _c * _h * _w * _d)


class NSF(BaseModel):
    def __init__(
        self,
        train_loader,
        val_loader=None,
        model_path=None,
        steps_per_level=10,
        levels=3,
        multi_scale=True,  # what is multiscale ?
        actnorm=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # Optimization
        batch_size=256,
        learning_rate=1e-4,
        cosine_annealing=False,
        eta_min=0.,
        warmup_fraction=0.,
        num_steps=100000,
        temperatures=None,

        # coupling_transform
        coupling_layer_type='rational_quadratic_spline',
        hidden_channels=256,
        use_resnet=False,
        num_res_blocks=5,  # If using resnet
        resnet_batchnorm=True,
        dropout_prob=0.,

        intervals=None,
        spline_parameters: dict = None,

    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.shape = next(iter(self.train_loader))['image'].shape
        print("input shape before passing to network: ", self.shape)

        # model params
        self.steps_per_level = steps_per_level
        self.levels = levels
        self.multi_scale = multi_scale
        self.actnorm = actnorm
        self.device = device

        # optimization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cosine_annealing = cosine_annealing
        self.eta_min = eta_min
        self.warmup_fraction = warmup_fraction
        self.num_steps = num_steps
        self.temperatures = [0.5, 0.75, 1.] if temperatures is None else temperatures

        # Coupling transform net
        self.coupling_layer_type = coupling_layer_type
        self.hidden_channels = hidden_channels
        self.use_resnet = use_resnet
        self.num_res_blocks = num_res_blocks  # If using resnet
        self.resnet_batchnorm = resnet_batchnorm
        self.dropout_prob = dropout_prob
        self.spline_parameters = spline_parameters

        self.flow_checkpoint = None
        self.optimizer_checkpoint = None
        self.start_step = 0
        self.intervals = intervals_ if intervals is None else intervals

        self.model = self.create_flow()
        super().__init__(self.model, model_path=model_path, base_filename="NSF", device=device)

    def create_flow(self):
        c, h, w, d = self.shape[1:]
        distribution = StandardNormal((c * h * w * d,))
        transform = self.create_transform(c, h, w, d)

        _flow = Flow(transform, distribution)

        if self.flow_checkpoint is not None:
            _flow.load_state_dict(torch.load(self.flow_checkpoint))

        return _flow

    def create_transform(self, c, h, w, d,):
        if not isinstance(self.hidden_channels, list):
            self.hidden_channels = [self.hidden_channels] * self.levels
            print("hidden channels, ", self.hidden_channels)

        if self.multi_scale:
            mct = MultiscaleCompositeTransform(num_transforms=self.levels)
            for level, level_hidden_channels in zip(range(self.levels), self.hidden_channels):
                squeeze_transform = SqueezeTransform()
                c, h, w, d = squeeze_transform.get_output_shape(c, h, w, d)
                print("channels: ", c)

                transform_step = [create_transform_step(c, level_hidden_channels, self.actnorm, self.spline_parameters,
                                                        self.coupling_layer_type, self.use_resnet, self.dropout_prob)
                                  for _ in range(self.steps_per_level)]

                transform_pipeline = [squeeze_transform]
                transform_pipeline.extend(transform_step)
                transform_pipeline.append(OneByOneConvolution(c))
                transform_level = CompositeTransform(transform_pipeline)

                new_shape = mct.add_transform(transform_level, (c, h, w, d))
                if new_shape:  # If not last layer
                    c, h, w, d = new_shape
        else:
            all_transforms = []

            for level, level_hidden_channels in zip(range(self.levels), self.hidden_channels):
                squeeze_transform = SqueezeTransform()
                c, h, w, d = squeeze_transform.get_output_shape(c, h, w, d)

                transform_step = [create_transform_step(c, level_hidden_channels, self.actnorm, self.spline_parameters,
                                                        self.coupling_layer_type, self.use_resnet, self.dropout_prob)
                                  for _ in range(self.steps_per_level)]

                transform_pipeline = [squeeze_transform]
                transform_pipeline.extend(transform_step)
                transform_pipeline.append(OneByOneConvolution(c))
                transform_level = CompositeTransform(transform_pipeline)
                all_transforms.append(transform_level)

            all_transforms.append(ReshapeTransform(
                input_shape=(c, h, w, d),
                output_shape=(c * h * w * d,)
            ))
            mct = CompositeTransform(all_transforms)

        # Inputs to the model in [0, 2 ** num_bits]
        return mct

    def train(self, **save_kwargs):
        c, h, w, d = self.shape[1:]
        self.model = self.model.to(self.device)

        # Random batch and identity transform for reconstruction evaluation.
        # random_batch = next(iter(self.train_loader))['image'] # .to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.optimizer_checkpoint is not None:
            optimizer.load_state_dict(torch.load(self.optimizer_checkpoint))

        if self.cosine_annealing:
            if self.warmup_fraction == 0.:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.num_steps,
                    last_epoch=-1 if self.start_step == 0 else self.start_step,
                    eta_min=self.eta_min
                )
            else:
                raise NotImplementedError
        else:
            scheduler = None

        best_val_log_prob = None
        start_time = None
        num_batches = self.num_steps - self.start_step
        print("number of batches: ", num_batches)

        progress_bar = tqdm(enumerate(load_batches(loader=self.train_loader, n_batches=num_batches)),
                            total=num_batches, ncols=110)
        progress_bar.set_description(f"Step ")

        for step, batch in progress_bar:
            if step == 0:
                start_time = time.time()  # Runtime estimate will be more accurate if set here.

            self.model.train()

            optimizer.zero_grad()

            batch = batch['image'].to(dtype=torch.float32, device=self.device)  # .to(self.device)

            assert batch.numel() > 0, "Empty batch encountered!"

            log_density = self.model.log_prob(batch)

            loss = -nats_to_bits_per_dim(torch.mean(log_density), c, h, w, d)
            progress_bar.set_postfix({"loss": loss})

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if step > 0 and step % self.intervals['eval'] == 0 and (self.val_loader is not None):
                val_log_prob = self.eval_log_density()
                val_log_prob = val_log_prob[0].item()  # mean loss for all validation batches

                if best_val_log_prob is None or val_log_prob > best_val_log_prob:
                    best_val_log_prob = val_log_prob
                    self.save(best_val="1", **save_kwargs)

    def eval_log_density(self, num_batches=None):
        c, h, w, d = self.shape[1:]
        with torch.no_grad():
            total_ld = []
            batch_counter = 0
            for _batch in self.val_loader:
                _batch = _batch['image'].to(dtype=torch.float32, device=self.device)
                log_prob = self.model.log_prob(_batch)
                total_ld.append(log_prob)
                batch_counter += 1
                if (num_batches is not None) and batch_counter == num_batches:
                    break
            total_ld = torch.cat(total_ld)
            total_ld = nats_to_bits_per_dim(total_ld, c, h, w, d)
            return total_ld.mean(), 2 * total_ld.std() / total_ld.shape[0]


if __name__ == "__main__":
    dataset = DataSet(**data_kwargs)
    train_loader, val_loader = get_train_val_loader(dataset, split_ratio=.9, **train_loader_kwargs)

    flow = NSF(train_loader, val_loader, **nsf_params)
    print("device: ", flow.device)
    flow.train()
