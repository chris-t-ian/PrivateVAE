import hashlib
import numpy as np


def check_and_remove_channel_dimension(x, dim=3):
    if dim == 3 and len(x.shape) == 5:
        assert x.shape[1] == 1, f"input file has {x.shape[1]} channels"
        return x[:, 0, :, :, :]
    if dim == 2 and len(x.shape) == 4:
        assert x.shape[1] == 1, f"input file has {x.shape[1]} channels"
        return x[:, 0, :, :]
    else:
        return x


def get3d_middle_slices(x, output="concatenated"):
    assert len(x.shape) == 3, f"dimensions of shape are {len(x.shape)}, but need 3"
    axial = x[:, :, x.shape[2] // 2]
    coronal = x[:, x.shape[1] // 2, :]
    sagittal = x[x.shape[0] // 2, :, :]
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
