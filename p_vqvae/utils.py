import hashlib
import numpy as np
import pandas as pd

def downsample_image(image: np.ndarray, factor):
    f = factor
    if len(image.shape) == 3:
        image = image[::f, ::f, ::f]
    elif len(image.shape) == 2:
        image = image[::f, ::f]
    elif len(image.shape) == 1:
        image = image[::f]
    else:
        raise ValueError(f"{len(image.shape)} dimensional input not supported.")

    return image


def check_and_remove_channel_dimension(x, dim=3):
    if dim == 3 and len(x.shape) == 5:
        assert x.shape[1] == 1, f"input file has {x.shape[1]} channels"
        return x[:, 0, :, :, :]
    if dim == 2 and len(x.shape) == 4:
        assert x.shape[1] == 1, f"input file has {x.shape[1]} channels"
        return x[:, 0, :, :]
    else:
        return x


def get3d_middle_slice(x, output="concatenated"):
    assert len(x.shape) == 3, f"dimensions of shape are {len(x.shape)}, but need 3"
    axial = x[:, :, x.shape[2] // 2]
    coronal = x[:, x.shape[1] // 2, :]
    sagittal = x[x.shape[0] // 2, :, :]
    if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0:
        print("Warning: x is empty in at least one dimension:", x.shape)

    if output == "concatenated":
        image_0 = np.concatenate([axial, np.flipud(coronal.T)], axis=1)
        image_1 = np.concatenate([np.flipud(sagittal.T), np.zeros((x.shape[0], x.shape[2]))], axis=1)
        return np.concatenate([image_0, image_1], axis=0)
    elif output == "axial":
        return axial
    elif output == "coronal":
        return coronal
    elif output == "sagittal":
        return sagittal
    else:
        raise NotImplementedError(f"Output {output} not implemented")


def get_2d_dataset_from_3d_dataset(x_3d, slice_type="axial", downsampling_factor=1):
    f = downsampling_factor

    if slice_type == "axial":
        data_2d = np.empty((x_3d.shape[0], x_3d.shape[1], x_3d.shape[2] // f, x_3d.shape[3] // f), dtype=x_3d.dtype)
    elif slice_type == "coronal":
        data_2d = np.empty((x_3d.shape[0], x_3d.shape[1], x_3d.shape[2] // f, x_3d.shape[4] // f), dtype=x_3d.dtype)
    elif slice_type == "sagittal":
        data_2d = np.empty((x_3d.shape[0], x_3d.shape[1], x_3d.shape[3] // f, x_3d.shape[4] // f), dtype=x_3d.dtype)
    else:
        raise NotImplementedError

    x_3d = check_and_remove_channel_dimension(x_3d, dim=3)

    # for every image in x_3d: load slice
    for idx in range(x_3d.shape[0]):
        image_3d = np.copy(x_3d[idx])
        image_2d = get3d_middle_slice(image_3d, output="axial")
        image_2d = downsample_image(image_2d, factor=f) if f > 1 else image_2d
        data_2d[idx, 0 , :, :] = image_2d  # add channel dimension

    return data_2d


def select_tpr_at_low_fprs(tprs, fprs, low_fpr: int = 0.01, reverse_order=True):
    _tprs = tprs.copy()
    _fprs = fprs.copy()
    if reverse_order:
        _tprs.reverse()
        _fprs.reverse()

    return _tprs[np.argmax(np.array(_fprs) <= low_fpr)] if np.any(np.array(_fprs) <= low_fpr) else _tprs[-1]

def subset_to_sha256_key(subset, len_set=956, len_key=8):
    """Create a unique compact hashed key for a subset of IDs ranging from 0 to len."""
    bit_vector = [0] * len_set
    for num in subset:
        bit_vector[num] = 1

    # Step 2: Convert to a binary string
    binary_str = ''.join(map(str, bit_vector))

    # Step 3: Hash the binary string to get a compact, unique key
    hashed_key = hashlib.sha256(binary_str.encode()).hexdigest()[:len_key]  # Take the first len_key chars for brevity
    return hashed_key


def calculate_AUC(tprs, fprs):
    return np.abs(np.trapz(tprs, x=fprs))


def get_auc_and_tpr_summary(df: pd.DataFrame):
    return pd.DataFrame({
        'mean_auc': df['auc'].mean(),
        'std_auc': df['auc'].std(),
        'mean_tpr_at_low_fpr': df['tpr_at_low_fpr'].mean(),
        'std_tpr_at_low_fpr': df['tpr_at_low_fpr'].std(),
        'low_fpr': df['low_fpr'].mean()
    }, index=[0])