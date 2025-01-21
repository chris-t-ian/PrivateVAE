import math
import time
import nflows
from monai.transforms import Compose
from torch.nn.functional import embedding
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
from nflows.utils import torchutils

import torch

from p_vqvae.dataloader import DataSet, get_train_val_loader, load_batches
from p_vqvae.networks import BaseModel, ResidualNet

device = "cuda:1"

# optimized for MRI data
data_kwargs = {
    "root": "data/ATLAS_2",
    "cache_path": 'data/cache/',
    "downsample": 4,
    "normalize": 1,
    "crop": ((8, 9), (12, 13), (0, 9)),
    "padding": ((1, 2), (0, 0), (1, 2)),
}
model_params = {
    "model_path": "model_outputs/mia",
    "device": device,
    "eval_interval": 2,  # after how many steps evaluate on validation set, before: 20
    "early_stopping": 5, #float('inf'),  # after how many evaluation steps stop the training
    "steps_per_level": 10,
    "levels": 2,  # increase for non-downsampled dataset
    "multi_scale": True,
    "actnorm": True,
}
_spline_params = {
        'num_bins': 4,  #
        'tail_bound': 1.,
        'min_bin_width': 1e-3,
        'min_bin_height': 1e-3,
        'min_derivative': 1e-3,
        'apply_unconditional_transform': False
}
optimization = {
    "epochs": 400, #400,
#   "batch_size": 16, #
    "learning_rate": 4e-4,
    "cosine_annealing": False,
    "eta_min": 0.,
#   "num_steps": 1000,  # change in implementation
    "mask_type": "alternating",
    "one_by_one_conv": True,
}
coupling_transform = {
    "coupling_layer_type": 'rational_quadratic_spline',
    "hidden_channels": 64,
    "use_resnet": False,
    "num_res_blocks": 5,  # If using resnet
    "resnet_batchnorm": True,
    "dropout_prob": 0.,
}

optimized_nsf_params = {**model_params, **optimization, **coupling_transform, "spline_parameters": _spline_params}


class Conv3dSameSize(torch.nn.Conv3d):
    """Makes sure that the output has the same shape as the input. Adaptation of Conv2dSameSize of
    nsf.experiments.autils for 3d data."""
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        # print("padding: ", same_padding)  # debugging print
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding)

class Conv2dSameSize(torch.nn.Conv2d):
    """Makes sure that the output has the same shape as the input. Adaptation of Conv2dSameSize of
    nsf.experiments.autils for 3d data."""
    def __init__(self, in_channels, out_channels, kernel_size):
        same_padding = kernel_size // 2  # Padding that would keep the spatial dims the same
        # print("padding: ", same_padding)  # debugging print
        super().__init__(in_channels, out_channels, kernel_size, padding=same_padding)

class ConvNet3D(torch.nn.Module):
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

class ConvNet2D(torch.nn.Module):
    """Encoder to reduce dimensions of MRI data."""
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.net = torch.nn.Sequential(
            Conv2dSameSize(in_channels, hidden_channels, kernel_size=3),
            torch.nn.ReLU(),
            Conv2dSameSize(hidden_channels, hidden_channels, kernel_size=1),
            torch.nn.ReLU(),
            Conv2dSameSize(hidden_channels, out_channels, kernel_size=3),
        )

    def forward(self, x, context=None):
        return self.net.forward(x)


class _CompositeTransform(CompositeTransform):
    @staticmethod
    def _cascade(inputs, funcs, context):
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            logabsdet = logabsdet.to(device=total_logabsdet.device)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet


def create_transform_step(num_channels, hidden_channels, actnorm, spline_params, coupling_layer_type,
                          use_resnet, dropout_prob, num_bins, one_b_one_conv=True, mask_type="mid_split", iteration=0,
                          spatial_dim=3):
    if use_resnet:
        raise NotImplementedError
    else:
        if dropout_prob != 0.:
            raise ValueError()

        if spatial_dim == 3:
            def create_convnet(in_channels, out_channels):
                return ConvNet3D(in_channels, out_channels, hidden_channels=hidden_channels)

        elif spatial_dim == 2:
            def create_convnet(in_channels, out_channels):
                return ConvNet2D(in_channels, out_channels, hidden_channels=hidden_channels)

        else:
            raise NotImplementedError

    if mask_type == "mid_split":
        mask = nflows.utils.create_mid_split_binary_mask(num_channels)
    elif mask_type == "alternating":
        mask = nflows.utils.create_alternating_binary_mask(num_channels, even=(iteration % 2 == 0))
    else:
        raise NotImplementedError(f"mask_type {mask_type} not supported.")

    if coupling_layer_type == 'rational_quadratic_spline':
        coupling_layer = PiecewiseRationalQuadraticCouplingTransform(
            mask=mask,
            transform_net_create_fn=create_convnet,
            tails='linear',
            tail_bound=spline_params['tail_bound'],
            num_bins=num_bins,
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

    if one_b_one_conv:
        step_transforms.append(OneByOneConvolution(num_channels))

    step_transforms.append(coupling_layer)

    return _CompositeTransform(step_transforms)


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


class SqueezeTransform2D(Transform):
    """A transformation defined for 3D image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.

    Implementation adapted from https://github.com/pclucas14/pytorch-glow,
    https://github.com/chaiyujin/glow-pytorch and https://github.com/bayesiains/nsf


    References:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    > C. Durkan et al., Neural Spline Flows, NeurIPS 2019.
    """
    def __init__(self, factor=2):
        super(SqueezeTransform2D, self).__init__()

        if not type(factor) == int or factor <= 1:
            raise ValueError('Factor must be an integer > 1.')

        self.factor = factor

    def get_output_shape(self, c, h, w):
        return (c * self.factor * self.factor,
                h // self.factor,
                w // self.factor)

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError('Expecting inputs with 4 dimensions')

        batch_size, c, h, w = inputs.size()

        if h % self.factor != 0 or w % self.factor != 0:
            raise ValueError('Input image size not compatible with the factor.')

        inputs = inputs.view(batch_size, c, h // self.factor, self.factor, w // self.factor,
                             self.factor)
        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        inputs = inputs.view(batch_size, c * self.factor * self.factor, h // self.factor,
                             w // self.factor)
        # print("inputs shape after squeezing: ", inputs.shape)
        return inputs, torch.zeros(batch_size)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError('Expecting inputs with 4 dimensions')

        batch_size, c, h, w = inputs.size()

        if c < 4 or c % 4 != 0:
            raise ValueError('Invalid number of channel dimensions.')

        inputs = inputs.view(batch_size, c // self.factor ** 2, self.factor, self.factor, h, w)
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        inputs = inputs.view(batch_size, c // self.factor ** 2, h * self.factor, w * self.factor)

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
            # print("std: ", std)
            # print("mu: ", mu)

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
            # print(self.log_scale.data)
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
        len_input_shape = len(inputs.shape)
        if len_input_shape == 5:
            b, c, h, w, d = inputs.shape
            inputs = inputs.permute(0, 2, 3, 4, 1).reshape(b * h * w * d, c)
        else:
            b, c, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 3, 1).reshape(b * h * w, c)

        if inverse:
            outputs, logabsdet = super().inverse(inputs)
        else:
            outputs, logabsdet = super().forward(inputs)

        if len_input_shape == 5:
            outputs = outputs.reshape(b, h, w, d, c).permute(0, 4, 1, 2, 3)
            logabsdet = logabsdet.reshape(b, h, w, d)
        else:
            outputs = outputs.reshape(b, h, w, c).permute(0, 3, 1, 2)
            logabsdet = logabsdet.reshape(b, h, w)

        return outputs, torchutils.sum_except_batch(logabsdet)

    def forward(self, inputs, context=None):
        if inputs.dim() not in [4, 5]:
            raise ValueError("Inputs must be a 4D or 5D tensor.")

        inputs, _ = self.permutation(inputs)

        return self._lu_forward_inverse(inputs, inverse=False)

    def inverse(self, inputs, context=None):
        if inputs.dim() not in [4, 5]:
            raise ValueError("Inputs must be a 4D or 5D tensor.")

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


def nats_to_bits_per_dim(nats, batch_shape):
    if len(batch_shape) == 4:
        _c, _h, _w, _d = batch_shape
        return nats / (math.log(2) * _c * _h * _w * _d)
    elif len(batch_shape) == 3:
        _c, _h, _w = batch_shape
        return nats / (math.log(2) * _c * _h * _w)
    else:
        raise NotImplementedError

class NSF(BaseModel):
    def __init__(
        self,
        _train_loader,
        _val_loader=None,
        model_path=None,
        steps_per_level=10,
        levels=3,
        multi_scale=True,  # what is multiscale ?
        actnorm=False,
        eval_interval=10,  # after how many steps evaluate on validation set
        early_stopping=3,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),

        # Optimization
        epochs = 100,
        learning_rate=1e-4,
        cosine_annealing=False,
        eta_min=0.,
        num_bins=4,
        num_steps=None,
        squeeze_factor: int = 2,

        # coupling_transform
        mask_type = "mid_split",
        coupling_layer_type='rational_quadratic_spline',
        one_by_one_conv = True,
        hidden_channels=256,
        use_resnet=False,
        num_res_blocks=5,  # If using resnet
        resnet_batchnorm=True,
        dropout_prob=0.,

        spline_parameters: dict = None,

        seed=None
    ):
        self.train_loader = _train_loader
        self.val_loader = _val_loader
        self.shape = self.get_shape()
        print("Input shape to base model: ", self.shape)
        self.spatial_dim = len(self.shape) - 2
        assert self.spatial_dim in [2, 3], f"This NSF only accepts 2 or 3 spatial dimensions but got {self.spatial_dim}"

        # model params
        self.steps_per_level = steps_per_level
        self.levels = levels
        self.multi_scale = multi_scale
        self.actnorm = actnorm
        self.device = device

        # optimization, training params
        self.epochs = epochs
        self.batch_size = _train_loader.batch_size
        self.max_num_steps = epochs * _train_loader.batch_size if num_steps is None else num_steps
        self.learning_rate = learning_rate
        self.cosine_annealing = cosine_annealing
        self.eval_interval = eval_interval
        self.early_stopping_patience = early_stopping
        self.eta_min = eta_min
        self.squeeze_factor = squeeze_factor

        # Coupling transform net
        self.mask_type = mask_type
        self.one_by_one_conv = one_by_one_conv
        self.num_bins = num_bins
        self.coupling_layer_type = coupling_layer_type
        self.hidden_channels = hidden_channels
        self.use_resnet = use_resnet
        self.num_res_blocks = num_res_blocks  # If using resnet
        self.resnet_batchnorm = resnet_batchnorm
        self.dropout_prob = dropout_prob
        self.spline_parameters = spline_parameters

        self.log_density_list = []
        self.val_log_density_list = []
        self.loss_list = []
        self.val_loss_list = []
        self.flow_checkpoint = None
        self.optimizer_checkpoint = None

        self.model = self.create_flow()
        super().__init__(self.model, model_path=model_path, base_filename="NSF", device=device, seed=seed)

    def get_shape(self):
        return next(iter(self.train_loader)).shape

    def create_flow(self):
        if self.spatial_dim == 3:
            c, h, w, d = self.shape[1:]
            distribution = StandardNormal((c * h * w * d,))
            transform = self.create_transform(c, h, w, d)
        else:
            c, h, w = self.shape[1:]
            distribution = StandardNormal((c * h * w,))
            transform = self.create_transform_2D(c, h, w)

        _flow = Flow(transform, distribution)

        if self.flow_checkpoint is not None:
            _flow.load_state_dict(torch.load(self.flow_checkpoint))

        return _flow

    def create_transform(self, c, h, w, d,):
        if not isinstance(self.hidden_channels, list):
            self.hidden_channels = [self.hidden_channels] * self.levels

        if self.multi_scale:
            mct = MultiscaleCompositeTransform(num_transforms=self.levels)
            for level, level_hidden_channels in zip(range(self.levels), self.hidden_channels):
                if self.squeeze_factor > 1:
                    squeeze_transform = SqueezeTransform()
                    c, h, w, d = squeeze_transform.get_output_shape(c, h, w, d)

                transform_step = [create_transform_step(c, level_hidden_channels, self.actnorm, self.spline_parameters,
                                                        self.coupling_layer_type, self.use_resnet, self.dropout_prob,
                                                        self.num_bins, self.one_by_one_conv, self.mask_type, step, 3)
                                  for step in range(self.steps_per_level)]

                transform_pipeline = [squeeze_transform] if self.squeeze_factor > 1 else []
                transform_pipeline.extend(transform_step)
                transform_pipeline.append(OneByOneConvolution(c))
                transform_level = _CompositeTransform(transform_pipeline)

                new_shape = mct.add_transform(transform_level, (c, h, w, d))
                if new_shape:  # If not last layer
                    c, h, w, d = new_shape
        else:
            all_transforms = []

            for level, level_hidden_channels in zip(range(self.levels), self.hidden_channels):
                if self.squeeze_factor > 1:
                    squeeze_transform = SqueezeTransform()
                    c, h, w, d = squeeze_transform.get_output_shape(c, h, w, d)

                transform_step = [create_transform_step(c, level_hidden_channels, self.actnorm, self.spline_parameters,
                                                        self.coupling_layer_type, self.use_resnet, self.dropout_prob,
                                                        self.num_bins, self.one_by_one_conv, self.mask_type, step, 3)
                                  for step in range(self.steps_per_level)]

                transform_pipeline = [squeeze_transform] if self.squeeze_factor > 1 else []
                transform_pipeline.extend(transform_step)
                transform_pipeline.append(OneByOneConvolution(c))
                transform_level = _CompositeTransform(transform_pipeline)
                all_transforms.append(transform_level)

            all_transforms.append(ReshapeTransform(
                input_shape=(c, h, w, d),
                output_shape=(c * h * w * d,)
            ))
            mct = _CompositeTransform(all_transforms)

        # Inputs to the model in [0, 2 ** num_bits]
        return mct

    def create_transform_2D(self, c, h, w):
        if not isinstance(self.hidden_channels, list):
            self.hidden_channels = [self.hidden_channels] * self.levels

        if self.multi_scale:
            mct = MultiscaleCompositeTransform(num_transforms=self.levels)
            for level, level_hidden_channels in zip(range(self.levels), self.hidden_channels):
                if self.squeeze_factor > 1:
                    squeeze_transform = SqueezeTransform2D()
                    c, h, w = squeeze_transform.get_output_shape(c, h, w)

                transform_step = [create_transform_step(c, level_hidden_channels, self.actnorm, self.spline_parameters,
                                                        self.coupling_layer_type, self.use_resnet, self.dropout_prob,
                                                        self.num_bins, self.one_by_one_conv, self.mask_type, step, 2)
                                  for step in range(self.steps_per_level)]

                transform_pipeline = [squeeze_transform] if self.squeeze_factor > 1 else []
                transform_pipeline.extend(transform_step)
                transform_pipeline.append(OneByOneConvolution(c))
                transform_level = _CompositeTransform(transform_pipeline)

                new_shape = mct.add_transform(transform_level, (c, h, w))
                if new_shape:  # If not last layer
                    c, h, w = new_shape
        else:
            all_transforms = []

            for level, level_hidden_channels in zip(range(self.levels), self.hidden_channels):
                if self.squeeze_factor > 1:
                    squeeze_transform = SqueezeTransform2D()
                    c, h, w = squeeze_transform.get_output_shape(c, h, w)

                transform_step = [create_transform_step(c, level_hidden_channels, self.actnorm, self.spline_parameters,
                                                        self.coupling_layer_type, self.use_resnet, self.dropout_prob,
                                                        self.num_bins, self.one_by_one_conv, self.mask_type, step, 2)
                                  for step in range(self.steps_per_level)]

                transform_pipeline = [squeeze_transform] if self.squeeze_factor > 1 else []
                transform_pipeline.extend(transform_step)
                transform_pipeline.append(OneByOneConvolution(c))
                transform_level = _CompositeTransform(transform_pipeline)
                all_transforms.append(transform_level)

            all_transforms.append(ReshapeTransform(
                input_shape=(c, h, w),
                output_shape=(c * h * w)
            ))
            mct = _CompositeTransform(all_transforms)

        # Inputs to the model in [0, 2 ** num_bits]
        return mct

    def train(self, clear_mem=False):
        evals_without_improvement = 0
        best_model_weights = None
        self.model = self.model.to(self.device)

        # Random batch and identity transform for reconstruction evaluation.
        # random_batch = next(iter(self.train_loader))['image'] # .to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.optimizer_checkpoint is not None:
            optimizer.load_state_dict(torch.load(self.optimizer_checkpoint))

        if self.cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.max_num_steps,
                last_epoch=-1,
                eta_min=self.eta_min
            )
        else:
            scheduler = None

        best_val_log_prob = None

        for epoch in range(self.epochs):
            self.model.train()

            epoch_log_density = []
            epoch_loss = []
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:

                optimizer.zero_grad()

                batch = batch.to(dtype=torch.float32, device=self.device)  # .to(self.device)
                assert batch.numel() > 0, "Empty batch encountered!"

                log_density = self.model.log_prob(batch)

                loss = -nats_to_bits_per_dim(torch.mean(log_density), self.shape[1:])  # why torch.mean?
                progress_bar.set_postfix({"loss": loss})

                epoch_loss.append(loss)
                epoch_log_density.append(float(torch.mean(log_density).cpu()))

                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            self.log_density_list.append(sum(epoch_log_density) / len(epoch_log_density))  # mean loss per epoch
            self.loss_list.append(sum(epoch_loss) / len(epoch_loss))

            if (epoch + 1) % self.eval_interval == 0 and (self.val_loader is not None):
                val_loss = self.val_eval_log_density()
                mean_val_loss = val_loss[0].item()  # mean loss for all validation batches
                print("mean val loss: ", mean_val_loss)

                if best_val_log_prob is None or mean_val_loss > best_val_log_prob:
                    best_val_log_prob = mean_val_loss
                    best_model_weights = self.model.state_dict()
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1

                # early stopping
                if evals_without_improvement >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}.")

                    if best_model_weights:
                        print("Loading best model weights so far.")
                        self.model.load_state_dict(best_model_weights)

                    break

        # self.save(best_val="1", **save_kwargs)
        if clear_mem and "cuda" in str(self.device):
            torch.cuda.empty_cache()
        return best_val_log_prob

    def val_eval_log_density(self, num_batches=None):
        with torch.no_grad():
            total_ld = []
            batch_counter = 0
            for val_step, _batch in enumerate(self.val_loader):
                _batch = _batch.to(dtype=torch.float32, device=self.device)
                log_prob = self.model.log_prob(_batch)
                total_ld.append(log_prob)
                batch_counter += 1
                if (num_batches is not None) and batch_counter == num_batches:
                    break
            total_ld = torch.cat(total_ld)
            self.val_log_density_list.append(torch.mean(total_ld).float().cpu())
            total_ld = nats_to_bits_per_dim(total_ld, self.shape[1:])
            self.val_loss_list.append(torch.mean(total_ld).float().cpu())
            return total_ld.mean(), 2 * total_ld.std() / total_ld.shape[0]

    def eval_log_density(self, input):
        assert input.dim() in [4, 5], "Give 4 or 5 dimensional input."
        with torch.no_grad():
            input = input.to(device=self.device, dtype=torch.float32)
            log_prob = self.model.log_prob(input)
            return log_prob

    def sample(self, n_samples):
        with torch.no_grad():
            samples = self.model.sample(n_samples, batch_size=self.batch_size).detach().cpu().numpy()

        return samples

class OptimizedNSF(NSF):
    def __init__(self, _train_loader, _val_loader=None, nsf_kwargs: dict = None):
        nsf_kwargs = optimized_nsf_params if nsf_kwargs is None else nsf_kwargs
        super().__init__(_train_loader, _val_loader, **nsf_kwargs)


class EmbeddingNSF(BaseModel):
    """Flatten Embedding and train NSF on embeddings."""
    def __init__(
            self,
            _train_loader,
            _val_loader=None,
            model_path=None,

            embedding_dim=64,

            num_flow_steps=8,
            levels=3,
            multi_scale=True,  # what is multiscale ?
            actnorm=False,
            eval_interval=10,  # after how many steps evaluate on validation set
            early_stopping=3,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),

            # Optimization
            epochs = 100,
            learning_rate=1e-4,
            cosine_annealing=False,
            eta_min=0.,
            num_bins=4,
            num_steps=None,

            # coupling_transform
            mask_type = "alternate",
            coupling_layer_type='rational_quadratic_spline',
            one_by_one_conv = False,
            hidden_channels=256,
            context_features=None,
            use_resnet=False,
            num_res_blocks=5,  # If using resnet
            resnet_activation=torch.nn.functional.relu,
            resnet_batchnorm=True,
            dropout_prob=0.,

            spline_parameters: dict = None,

            seed=None

    ):
        self.embedding_dim = embedding_dim

        # flow params
        self.num_flow_steps = num_flow_steps

        # coupling params
        self.mask_type = mask_type
        assert coupling_layer_type in ["rational_quadratic_spline"]
        self.num_bins = num_bins

        # resnet params:
        self.hidden_channels = hidden_channels
        self.context_features = context_features
        self.num_res_blocks = num_res_blocks
        self.resnet_activation = resnet_activation
        self.dropout_prob = dropout_prob
        self.resnet_batchnorm = resnet_batchnorm

        print("embedding_shape (should be (batch_size, 64): ",  next(iter(self.train_loader)).shape)
        super().__init__(self.model, model_path=model_path, base_filename="NSF_latent", device=device, seed=seed)

    def get_shape(self):
        return next(iter(self.train_loader)).shape

    def create_flow(self):
        base_distribution = StandardNormal([self.embedding_dim])
        transform = self.create_transform()

        _flow = Flow(transform, base_distribution)

        if self.flow_checkpoint is not None:
            _flow.load_state_dict(torch.load(self.flow_checkpoint))

        return _flow

    def create_transform(self):
        # linear "lu" transform
        transform_steps = [
            nflows.transforms.RandomPermutation(self.latent_dim),
            nflows.transforms.LULinear(self.latent_dim, identity_init=True)
        ]

        # add spline transforms
        transform_steps.extend([self._create_transform_step(i) for i in range(self.num_flow_steps)])

        return CompositeTransform(transform_steps)


    def _create_transform_step(self, iteration):
        _transform = _PiecewiseRationalQuadraticCouplingTransform(
            mask = self.get_mask(iteration),
            transform_net_create_fn = lambda in_features,
                                             out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=self.hidden_channels,
                context_features=self.context_features,
                num_blocks=self.num_res_blocks,
                activation=self.resnet_activation,
                dropout_probability=self.dropout_prob,
                use_batch_norm=self.resnet_batchnorm
            ),
            num_bins=self.num_bins,
            tails='linear',
        )
        return _transform

    def get_mask(self, iteration):
        if self.mask_type=="alternate":
            return nflows.utils.create_alternating_binary_mask(self.embedding_dim, even=(iteration % 2==0))
        elif self.mask_type=="mid_split":
            return nflows.utils.create_alternating_binary_mask(self.embedding_dim)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    dataset = DataSet(**data_kwargs)
    # train_loader, val_loader = get_train_val_loader(dataset, split_ratio=.9, **train_loader_kwargs)

    #flow = NSF(train_loader, val_loader, **optimized_nsf_params)
    # print("device: ", flow.device)
    # flow.train()
