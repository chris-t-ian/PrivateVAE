from torchmetrics.image.fid import FrechetInceptionDistance
#from pytorch_msssim import ms_ssim
from torch import from_numpy
from MedicalNet.model import generate_model, MedNetOptions
import torch
import numpy as np


def check_dimension_and_type(real_images, generated_images, dtype=torch.uint8):
    if len(generated_images.shape) == 4:
        generated_images = np.expand_dims(generated_images, axis=1)
    assert len(real_images.shape) == len(
        generated_images.shape), "real_images and generated_images must have the same number of dimensions"

    if real_images.dtype != np.dtype('uint8') and real_images.max() < 1.01 and dtype == torch.uint8:
        real_images = real_images * 255.0
        generated_images = generated_images * 255.0

    return from_numpy(real_images).to(dtype=dtype), from_numpy(generated_images).to(dtype=dtype)


def calculate_FMD_3D(real_images, generated_images, device=None):
    """
    Calculates "Fréchet MedicalNet distance" between real and generated 3-dimensional MRI data. For a working
    implementation flatten the output of MedicalNet:

    class ResNet(nn.Module):
        def forward(self, x):
            ...
            x = torch.flatten(x, 1)  # add this line
            """
    real_images, generated_images = check_dimension_and_type(real_images, generated_images, dtype=torch.float32)
    input_shape = real_images.shape[1:]

    opt = MedNetOptions(
        model='resnet',
        model_depth=34,
        input_shape=input_shape[1:],
        use_gpu=True,
        gpu_id=[0],
        n_seg_classes=1,  # Number of segmentation classes?
        resnet_shortcut_type='B'
    )
    med3d_resnet, med_parameters = generate_model(opt)

    fid_metric = FrechetInceptionDistance(
        feature=med3d_resnet,
        normalize=True,
        input_img_size=input_shape)

    if device:
        fid_metric = fid_metric.to(device)
        real_images = real_images.to(device)
        generated_images = generated_images.to(device)

    fid_metric.update(real_images, real=True)
    fid_metric.update(generated_images, real=False)

    return fid_metric.compute().item()


def calculate_FID_2D(real_images, generated_images, n_images=30):
    """Calculates FID scores for axial-, coronal- and saggital-middle slices.
  Returns FID for each slice."""
    real_images, generated_images = check_dimension_and_type(real_images, generated_images)

    n_images = min(n_images, real_images.shape[0])
    n_images = min(n_images, generated_images.shape[0])

    real_images = real_images[:n_images]
    generated_images = generated_images[:n_images]

    real_images = from_numpy(real_images).byte()  # convert T1_data to torch tensor
    real_images = np.repeat(real_images, 3, axis=1)  # repeat greyscale channel to 3 channels

    real_images_slice1 = real_images[:, :, real_images.shape[2] // 2, :, :]  # take middle slice of each image
    real_images_slice2 = real_images[:, :, :, real_images.shape[3] // 2, :]  # take middle slice of each image
    real_images_slice3 = real_images[:, :, :, :, real_images.shape[4] // 2]  # take middle slice of each image

    generated_images = from_numpy(generated_images)  # convert T1_data to torch tensor
    generated_images = generated_images.repeat(1, 3, 1, 1, 1)  # repeat greyscale channel to 3 channels
    # generated_images_with_channels = np.zeros((generated_images.shape[0], 3,
    #                                      generated_images.shape[2],
    #                                      generated_images.shape[3],
    #                                      generated_images.shape[4]), dtype=np.uint8)
    # generated_images_with_channels[:, 0, :, :, :] = generated_images[:, 0, :, :, :]
    # generated_images = generated_images_with_channels

    generated_images_slice1 = generated_images[:, :, generated_images.shape[2] // 2, :, :]  # take middle slice
    generated_images_slice2 = generated_images[:, :, :, generated_images.shape[3] // 2, :]  # take middle slice
    generated_images_slice3 = generated_images[:, :, :, :, generated_images.shape[4] // 2]  # take middle slice

    fid_metric_slice1 = FrechetInceptionDistance(feature=2048)
    fid_metric_slice2 = FrechetInceptionDistance(feature=2048)
    fid_metric_slice3 = FrechetInceptionDistance(feature=2048)

    fid_metric_slice1.update(real_images_slice1, real=True)
    fid_metric_slice2.update(real_images_slice2, real=True)
    fid_metric_slice3.update(real_images_slice3, real=True)

    fid_metric_slice1.update(generated_images_slice1, real=False)
    fid_metric_slice2.update(generated_images_slice2, real=False)
    fid_metric_slice3.update(generated_images_slice3, real=False)

    fid_score_slice1 = fid_metric_slice1.compute()
    fid_score_slice2 = fid_metric_slice2.compute()
    fid_score_slice3 = fid_metric_slice3.compute()

    return float(fid_score_slice1), float(fid_score_slice2), float(fid_score_slice3)


def calculate_msssim(real_data, synthetic_data, win_size: int):
    """Calculate ms-ssim score between real and reconstructed images."""
    real_data = real_data if torch.is_tensor(real_data) else from_numpy(real_data)
    synthetic_data = synthetic_data if torch.is_tensor(synthetic_data) else from_numpy(synthetic_data)

    #return ms_ssim(real_data, synthetic_data, data_range=1, size_average=True, win_size=win_size)
