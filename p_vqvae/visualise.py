import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, get_device
from p_vqvae.utils import calculate_AUC


def check_and_remove_channel_dimension(x):
    if len(x.shape) == 5:
        assert x.shape[1] == 1, f"input file has {x.shape[1]} channels"
        return x[:, 0, :, :, :]
    else:
        return x


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

    if n==1:
        reconstruction = get3d_middle_slices(reconstructions[0])

        ax[0].imshow(reconstruction, cmap="gray")
        ax[0].axis("off")
        ax[0].title.set_text(f"Reconstructed image")

        image = get3d_middle_slices(img[0])

        ax[1].imshow(image, cmap="gray")
        ax[1].axis("off")
        ax[1].title.set_text(f"Real image")
    else:
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


def plot_real_rec_syn(img: np.array, reconstructions: np.array, synthetic: np.array, k=0):
    img = check_and_remove_channel_dimension(img)
    reconstructions = check_and_remove_channel_dimension(reconstructions)
    synthetic = check_and_remove_channel_dimension(synthetic)

    assert img.shape == reconstructions.shape, f"shape mismatch between inputs {img.shape} vs {reconstructions.shape}"
    assert img.shape == synthetic.shape, f"shape mismatch between inputs {img.shape} vs {synthetic.shape}"

    fig, ax = plt.subplots(nrows=3, ncols=1)
    plt.style.use("default")

    titles = ["Real image", "Reconstruction", "Sampled"]
    images = [img, reconstructions, synthetic]

    for i, y in enumerate(images):
        x = np.concatenate([
            y[k, :, :, y.shape[3]//2],
            np.flipud(y[k, :, y.shape[2]//2, :].T),
            np.flipud(y[k, y.shape[1]//2, :, :].T),
        ], axis=1)
        ax[i].imshow(x, cmap="gray")
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_ylabel(titles[i])

    plt.show()


def show_roc_curve(tprs, fprs, label=None, tprs2=None, fprs2=None, label2=None, tprs3=None, fprs3=None, label3=None):
    auc = calculate_AUC(tprs, fprs)
    if label:
        plt.plot(fprs, tprs, marker='.', label=f'{label} (AUC = {auc:.4f})')
    else:
        plt.plot(fprs, tprs, marker='.', label=f'AUC = {auc:.4f}')

    if tprs2 and fprs2:
        assert label and label2, "Specify labels."
        auc = calculate_AUC(tprs2, fprs2)
        plt.plot(fprs2, tprs2, marker='.', label=f'{label2} (AUC = {auc:.4f})')

    if tprs3 and fprs3:
        assert label3, "Specify label 3"
        auc = calculate_AUC(tprs3, fprs3)
        plt.plot(fprs3, tprs3, marker='.', label=f'{label3} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random chance
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

