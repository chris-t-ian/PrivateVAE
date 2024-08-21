import os
import numpy as np

import nibabel as nib
from monai.data import DataLoader
from tqdm import tqdm


def transform(data: np.array, downsample_factor=1, normalize=True, crop_size=4, padding=True) -> np.array:
    """
  Transforms a 3D MRI image using the specified parameters.
  """
    if padding:
        data = np.pad(data, ((1, 2), (0, 0), (5, 6)), mode='constant')
    if downsample_factor != 1:
        f = downsample_factor
        data = data[::f, ::f, ::f]  # downsample image, take every f-th element
    if normalize:
        data = (data - data.min()) / (data.max() - data.min())
    if crop_size != 0:  # crop image
        assert crop_size >= 0
        data = data[crop_size:-crop_size, crop_size:-crop_size - 1, crop_size:-crop_size]
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
            downsample: int = 1,
            normalize: bool = True,
            crop_size: int = 4,
            padding: bool = True
    ):
        self.mode = mode
        self.downsample = downsample
        self.prefix = f"_{mode}_smpl{downsample}"
        self.normalize = normalize
        self.crop_size = crop_size
        self.padding = padding

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
        n = 32 if self.mode == "tiny" else 10000  # number of images to load
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
            data = []

            # Load images using memory mapping
            for img_path in paths:
                img = nib.load(img_path, mmap=True)  # Use memory mapping
                img_data = img.get_fdata()  # This will not load the entire file into memory
                img_data = transform(img_data, self.downsample, self.normalize, self.crop_size, self.padding)
                data.append(img_data)
                progress_bar.update(1)
            data = np.stack(data)

            progress_bar.close()

            data = np.expand_dims(data, axis=1)  # add channel dimension

            if cache_path:
                print(f"Storing transformed data as {cached_file}")
                np.save(cached_file, data)

        return data


class TinyDataSet(DataSet):
    def __int__(self):
        self.mode = "tiny"