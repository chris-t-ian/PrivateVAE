import os
import numpy as np
import torch

import nibabel as nib
from monai.data import DataLoader
from monai.data import Dataset as DataSet_monai
from monai.transforms import Compose, RandAffine, RandShiftIntensity, RandGaussianNoise, ThresholdIntensity, ToTensor
from torch.utils.data import random_split
from tqdm import tqdm


def preprocess(data: np.array, downsample_factor: int = 1, normalize: int = None,
              crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None,
              padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None) -> np.array:
    """
    Transforms a 3D MRI image using the specified parameters.
    """
    if crop:  # crop image
        data = data[crop[0][0]:-crop[0][1], crop[1][0]:-crop[1][1], crop[2][0]:-crop[2][1]]
    if downsample_factor != 1:  # down-sample image, take every f-th element
        f = downsample_factor
        data = data[::f, ::f, ::f]
    if normalize:  # normalize image to range [0, normalize]
        data = (data - data.min()) / (data.max() - data.min()) * normalize
    if padding:  #
        data = np.pad(data, padding, mode='constant')
    return data


def get_augmentation():
    transforms = Compose([
        # Step 1: Random Affine transformation
        RandAffine(
            prob=0.8,
            rotate_range=(0.04, 0.04),  # Rotation range in radians
            translate_range=(2, 2),  # Translation range in pixels
            scale_range=(0.05, 0.05),  # Scaling range
        ),

        # Step 2: Random Intensity Shift
        RandShiftIntensity(
            offsets=0.05,  # Max offset for intensity shift
            prob=0.3
        ),

        # Step 3: Random Gaussian Noise
        RandGaussianNoise(
            prob=0.6,
            mean=0.0,
            std=0.02
        ),

        # Step 4: Thresholding to ensure values are within [0, 1.0]
        ThresholdIntensity(
            threshold=0.0,  # Min value
            above=True,  # Values above the threshold will be set to 0
            cval=0.0,  # Clipping value
        ),
        ThresholdIntensity(
            threshold=1.0,  # Max value
            above=False,  # Values above the threshold will be set to 1
            cval=1.0  # Clipping value
        ),

        ToTensor()

    ])
    return transforms


def load_batches(loader, n_batches):
    """Return n_batches of batches given loader."""
    ii = 0
    while True:
        for batch in loader:
            yield batch
            ii += 1
            if ii == n_batches:
                return


def float32_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor from float32 to uint8.
    Assumes input tensor is in range [0, 1].
    """
    return (tensor * 255).clamp(0, 255).to(torch.uint8)


def uint8_to_float32(tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts a tensor from uint8 to float32.
    Output will be in the range [0, 1].
    """
    return tensor.to(torch.float32) / 255.0


class DataSet(DataSet_monai):
    """
    A generic dataset tailored to ATLAS v2.

    args:
        mode: Size of the dataset. Options: "full" loads all images, "tiny" loads 32 images
        downsample: downsampling factor
    """

    def __init__(
            self,
            data,
            dtype: np.dtype = np.float16,
    ):

        self.dtype = dtype
        self.transform = get_augmentation()
        self.data = data
        super().__init__(self.data)

    # overriding __len__ and __getitem__ methods of DataSet_monai
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.copy(self.data[idx])

        if self.transform:
            image = self.transform(image)

        if self.dtype == torch.float32 and image.dtype == torch.uint8:
            image = uint8_to_float32(image)
        elif self.dtype == torch.uint8 and image.dtype == torch.float32:
            image = float32_to_uint8(image)
        
        return {"image": image}

    def get_images(self, range_indices: list):
        assert len(range_indices) == 2, "Provided range indices are not a list of two values"

        n_images = range_indices[1] - range_indices[0]

        shape = (n_images, self.data.shape[1], self.data.shape[2], self.data.shape[3], self.data.shape[4])
        images = np.empty(shape, dtype=self.dtype)

        for i in range(n_images):
            img = np.copy(self.data[range_indices[0] + i])
            images[i] = img
        return images

    def get_data(self):
        return self.data

    def __load__(self):
        NotImplementedError("Loading of data is not implemented.")


class RawDataSet(DataSet):
    def __init__(
        self,
        mode: str = "full",
        root: str = os.getcwd(),
        cache_path: str = None,
        downsample: int = 1,
        normalize: int = 1,
        dtype: np.dtype = np.float16,
        crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None,
        padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None,
    ):
        self.mode = mode
        self.root = root
        self.cache_path = cache_path
        self.downsample = downsample
        self.prefix = f"_{mode}_smpl{downsample}"
        self.normalize = normalize
        self.crop_size = crop
        self.padding = padding
        self.dtype = dtype
        self.transform = get_augmentation()
        self.data = self.__load__()

        super().__init__(self.data, dtype)

    def __load__(self) -> np.array:
        """
        Load data from parent_path and apply transforms. If initiated for the first time load all T1 files.

        :arg
            parent_path: path were T1.nii.gz images are stored
            cache_path: creates a stored .npy file for faster loading time at the specified path
        """

        cached_file = os.path.join(self.cache_path, 'ATLAS_2' + self.prefix + '.npy')
        if os.path.isfile(cached_file) and self.cache_path:  # if file exists and cache path provided, load npy file
            data = np.load(cached_file, mmap_mode='r')
        else:
            paths = self.collect_paths(self.root)
            progress_bar = tqdm(total=len(paths), ncols=110, desc="Loading T1 images")  # initialise progress bar

            # load first image to define shape
            img = nib.load(paths[0], mmap=True)  # Use memory mapping
            img = img.get_fdata()  # This will not load the entire file into memory
            img = preprocess(img, self.downsample, self.normalize, self.crop_size, self.padding)
            shape = (len(paths), 1, img.shape[0], img.shape[1], img.shape[2])
            data = np.empty(shape, dtype=self.dtype)

            # Load images using memory mapping
            for k, img_path in enumerate(paths):
                img = nib.load(img_path, mmap=True)  # Use memory mapping
                img = img.get_fdata().astype(self.dtype)  # This will not load the entire file into memory
                img = preprocess(img, self.downsample, self.normalize, self.crop_size, self.padding)
                data[k, 0, :, :, :] = img[:, :, :]  # add channel dimension
                progress_bar.update(1)

            progress_bar.close()

            if self.cache_path:
                print(f"Storing transformed data as {cached_file}")
                np.save(cached_file, data)

        return data

    def collect_paths(
            self,
            parent_path: str = os.getcwd(),
            verbose: bool = True) -> list:
        """
        Collect image paths of files ending in 'T1w.nii.gz'

        :args
            parent_path: parent path to dataset containing .nii.gz files. Default: current working directory
        :returns
            list of image paths
        """
        # number of images to load:
        if self.mode == "tiny":
            n = 32
        elif self.mode == "small":
            n = 64
        elif self.mode == "third":
            n = 318
        elif self.mode == "half":
            n = 477
        else:
            n = 10000

        k = 0  # image counter
        img_paths = []

        for dirpath, _, files in os.walk(parent_path):
            for file in files:
                if file.endswith('T1w.nii.gz'):
                    k += 1
                    img_paths.append(os.path.join(dirpath, file))
                if k >= n:
                    break
            if k >= n:
                break

        return img_paths


class SyntheticDataSet(DataSet):
    def __init__(self, cached_file, labels=None, dtype=np.float16):
        self.cached_file = cached_file
        self.labels = labels
        self.images = self.__load__()
        super().__init__(self.images, dtype)

    def __load__(self):
        assert os.path.isfile(self.cached_file), f"Cached file {self.cached_file} does not exist."
        return np.load(self.cached_file, mmap_mode='r')

    def __getitem__(self, idx):
        image = np.copy(self.images[idx])

        if self.transform:
            image = self.transform(image)

        if self.dtype == torch.float32 and image.dtype == torch.uint8:
            image = uint8_to_float32(image)
        elif self.dtype == torch.uint8 and image.dtype == torch.float32:
            image = float32_to_uint8(image)

        if self.labels is not None:
            return {"image": image, "label": self.labels[idx]}
        else:
            return {"image": image}


def get_train_val_loader(
        dataset,
        batch_size: int = 8,
        augment_flag=True,
        split_ratio: float = 0.875,
        num_workers=8,
        ):

    transform = get_augmentation() if augment_flag else ToTensor()

    if split_ratio == 1.0:
        dataset.transform = transform

        # DataLoader calls __getitem__ and therefore performs the transform/augmentation on the fly
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers), None
    else:
        train_size = int(len(dataset) * split_ratio)
        test_size = len(dataset) - train_size

        train_set, test_set = random_split(dataset, [train_size, test_size])

        train_set.dataset.transform = transform if transform else ToTensor()
        test_set.dataset.transform = ToTensor()  # no augmentation for test data

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader


def get_train_loader(
    dataset,
    batch_size: int = 8,
    augment_flag=True,
    num_workers=8,
):
    transform = get_augmentation() if augment_flag else ToTensor()
    dataset.transform = transform

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
