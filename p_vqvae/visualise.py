import os.path

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, get_device
from p_vqvae.utils import calculate_AUC, get3d_middle_slices, check_and_remove_channel_dimension

plt.rcParams['figure.figsize'] = [5, 3.75]


def plot_generated_images(img: np.array, n=3, file="data/plots/generated_images.png", dim=3):

    img = check_and_remove_channel_dimension(img, dim)
    if n == 1:
        image = get3d_middle_slices(img[0]) if dim == 3 else img[1]
        plt.imshow(image, cmap="gray")

    else:
        fig, ax = plt.subplots(nrows=1, ncols=n)
        plt.style.use("default")

        for i in range(n):
            image = get3d_middle_slices(img[i])
            ax[i].imshow(image, cmap="gray")
            ax[i].axis("off")
            ax[i].title.set_text(f"Synthetic image {i}")

    plt.savefig(file)
    plt.clf()


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

    plt.savefig("data/plots/reconstructions.png")
    plt.clf()


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


def show_roc_curve(tprs, fprs, label=None, tprs2=None, fprs2=None, label2=None, tprs3=None, fprs3=None, label3=None,
                   low_fprs=False, save=True):
    auc = calculate_AUC(tprs, fprs)
    if label:
        plt.plot(fprs, tprs, label=f'{label} (AUC = {auc:.3f})')
    else:
        plt.plot(fprs, tprs, label=f'AUC = {auc:.3f}')

    if tprs2 is not None and fprs2 is not None:
        assert label and label2, "Specify labels."
        auc = calculate_AUC(tprs2, fprs2)
        plt.plot(fprs2, tprs2, label=f'{label2} (AUC = {auc:.3f})')

    if tprs3 is not None and fprs3 is not None:
        assert label3, "Specify label 3"
        auc = calculate_AUC(tprs3, fprs3)
        plt.plot(fprs3, tprs3, label=f'{label3} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random chance
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    if low_fprs:
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(1e-3, 1.0)
        plt.ylim(1e-3, 1.0)

    plt.subplots_adjust(left=0.2, bottom=0.2)
    file_path = f"data/plots/ROC_{label}_{label2}_{label3}_lowfprs{low_fprs}.png"
    plt.grid()
    if save:
        print("saving plot as ", file_path)
        plt.savefig(file_path)
        plt.clf()

def show_roc_curve_std(std1, std2=None, std3=None, file_path="data/plots", **kwargs):
    show_roc_curve(**kwargs, save=False)
    tprs = kwargs["tprs"]

    plt.fill_between(kwargs["fprs"], tprs + std1, tprs - std1, alpha=0.3, color='#888888')
    if std2 is not None:
        tprs2 = kwargs["tprs2"]
        plt.fill_between(kwargs["fprs2"], tprs2 + std2, tprs2 - std2, alpha=0.3, color='#888888')
    if std3 is not None:
        tprs3 = kwargs["tprs3"]
        plt.fill_between(kwargs["fprs3"], tprs3 + std3, tprs3 - std3, alpha=0.3, color='#888888')

    log_scale_label = "log" if kwargs.get("low_fprs") else ""
    file_path = os.path.join(file_path, f"ROC_std_{kwargs['label']}_{log_scale_label}.png")
    plt.savefig(file_path)
    plt.clf()

def show_auc_tpr_plot(x_values, mean_auc, std_auc=None, xlabel=" ", ylabel="AUC"):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6, 6)
    ax.plot(x_values, mean_auc, yerr=std_auc, color="blue", ecolor='#888888')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.savefig(f"data/plots/{xlabel}_{ylabel}.png")
    plt.clf()