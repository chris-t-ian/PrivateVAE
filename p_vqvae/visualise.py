import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, get_device


def check_and_remove_channel_dimension(x):
    if len(x.shape) == 5:
        assert x.shape[1] == 1, f"input file has {x.shape[1]} channels"
        return x[:, 0, :, :, :]


def get3d_middle_slices(x):
    assert len(x.shape) == 3, f"dimensions of shape are {len(x.shape)}, but need 3"
    image_0 = np.concatenate(
        [x[:, :, x.shape[2] // 2], np.flipud(x[:, x.shape[1] // 2, :].T)],
        axis=1)
    image_1 = np.concatenate(
        [np.flipud(x[x.shape[0] // 2, :, :].T), np.zeros((x.shape[0], x.shape[2]))], axis=1)
    return np.concatenate([image_0, image_1], axis=0)


def plot_generated_images(img: np.array, n=3):

    img = check_and_remove_channel_dimension(img)

    fig, ax = plt.subplots(nrows=1, ncols=n)
    plt.style.use("default")

    for i in range(n):
        image = get3d_middle_slices(img[i])
        ax[i].imshow(image, cmap="gray")
        ax[i].axis("off")
        ax[i].title.set_text(f"Synthetic image {i}")

    plt.show()


def plot_reconstructions(img: np.array, reconstructions: np.array, n=1):

    img = check_and_remove_channel_dimension(img)
    reconstructions = check_and_remove_channel_dimension(reconstructions)
    assert img.shape == reconstructions.shape, f"shape mismatch between inputs {img.shape} vs {reconstructions.shape}"

    fig, ax = plt.subplots(nrows=n, ncols=2)
    plt.style.use("default")

    for i in range(n):
        reconstruction = get3d_middle_slices(reconstructions[i])

        ax[0, i].imshow(reconstruction, cmap="gray")
        ax[0, i].axis("off")
        ax[0, i].title.set_text(f"Reconstructed image {i}")

        image = get3d_middle_slices(img[i])

        ax[1, i].imshow(image, cmap="gray")
        ax[1, i].axis("off")
        ax[1, i].title.set_text(f"Real image {i}")

    plt.show()
