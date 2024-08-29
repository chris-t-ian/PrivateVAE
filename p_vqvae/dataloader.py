import os
import numpy as np

import nibabel as nib
from monai.data import DataLoader
from tqdm import tqdm


def transform(data: np.array, downsample_factor: int = 1, normalize: bool = True,
              crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None,
              padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None) -> np.array:
    """
    Transforms a 3D MRI image using the specified parameters.
    """
    if crop:  # crop image
        data = data[crop[0][0]:-crop[0][1], crop[1][0]:-crop[1][1], crop[2][0]:-crop[2][1]]
    if downsample_factor != 1:
        f = downsample_factor
        data = data[::f, ::f, ::f]  # downsample image, take every f-th element
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())
    if padding:
        data = np.pad(data, padding, mode='constant')
    return data


class DataSet:
    """
    A generic dataset tailored to ATLAS v2.

    args:
        mode: Size of the dataset. Options: "full" loads all images, "tiny" loads 32 images
        downsample: downsampling factor
    """

    def __init__(
            self,
            mode: str = "full",
            root: str = None,
            cache_path: str = None,
            downsample: int = 1,
            normalize: bool = True,
            crop: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None,
            padding: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] = None
    ):
        self.mode = mode
        self.downsample = downsample
        self.prefix = f"_{mode}_smpl{downsample}"
        self.normalize = normalize
        self.crop_size = crop
        self.padding = padding

        self.data = self.load(root, cache_path)

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

    def load(self, parent_path: str, cache_path: str) -> np.array:
        """
        Load data from parent_path and apply transforms. If initiated for the first time load all T1 files.

        :arg
            parent_path: path were T1.nii.gz images are stored
            cache_path: creates a stored .npy file for faster loading time at the specified path
        """

        cached_file = os.path.join(cache_path, 'ATLAS_2' + self.prefix + '.npy')
        if os.path.isfile(cached_file) and cache_path:  # if file exists and cache path provided, load npy file
            data = np.load(cached_file)
        else:
            paths = self.collect_paths(parent_path)
            progress_bar = tqdm(total=len(paths), ncols=110, desc="Loading T1 images")  # initialise progress bar

            # load first image to define shape
            img = nib.load(paths[0], mmap=True)  # Use memory mapping
            img = img.get_fdata()  # This will not load the entire file into memory
            img = transform(img, self.downsample, self.normalize, self.crop_size, self.padding)
            shape = (len(paths), 1, img.shape[0], img.shape[1], img.shape[2])
            data = np.empty(shape, dtype=np.float16)

            # Load images using memory mapping
            for k, img_path in enumerate(paths):
                img = nib.load(img_path, mmap=True)  # Use memory mapping
                img = img.get_fdata().astype(np.float16)  # This will not load the entire file into memory
                img = transform(img, self.downsample, self.normalize, self.crop_size, self.padding)
                data[k, 0, :, :, :] = img[:, :, :]
                progress_bar.update(1)

            progress_bar.close()

            if cache_path:
                print(f"Storing transformed data as {cached_file}")
                np.save(cached_file, data)

        return data

    def get_train_val_loader(self, batch_size, split_ratio=0.9):
        if split_ratio == 1.0:
            return DataLoader(self.data, batch_size=batch_size, shuffle=True)
        else:
            n_train = int(self.data.shape[0] * 0.9)
            train_loader = DataLoader(self.data[:n_train, ...], batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(self.data[n_train:, ...], batch_size=batch_size, shuffle=False)
            return train_loader, val_loader

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data(self):
        return self.data
