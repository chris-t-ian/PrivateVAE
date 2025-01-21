import os.path

import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, get_device
from p_vqvae.utils import calculate_AUC, get3d_middle_slice, check_and_remove_channel_dimension, select_tpr_at_low_fprs

plt.rcParams['figure.figsize'] = [5, 3.75]


def plot_generated_images(img: np.array, n=3, file="data/plots/generated_images.png", dim=3):

    img = check_and_remove_channel_dimension(img, dim)
    if n == 1:
        image = get3d_middle_slice(img[0]) if dim == 3 else img[0]
        plt.imshow(image, cmap="gray")

    else:
        fig, ax = plt.subplots(nrows=1, ncols=n)
        plt.style.use("default")

        for i in range(n):
            image = get3d_middle_slice(img[i]) if dim == 3 else img[i]
            ax[i].imshow(image, cmap="gray")
            ax[i].axis("off")
            ax[i].title.set_text(f"Sample {i}")

    plt.savefig(file)
    plt.clf()


def plot_reconstructions(img: np.array, reconstructions: np.array, n=1):

    img = check_and_remove_channel_dimension(img)
    reconstructions = check_and_remove_channel_dimension(reconstructions)
    assert img.shape == reconstructions.shape, f"shape mismatch between inputs {img.shape} vs {reconstructions.shape}"

    fig, ax = plt.subplots(nrows=n, ncols=2)
    plt.style.use("default")

    if n==1:
        reconstruction = get3d_middle_slice(reconstructions[0])

        ax[0].imshow(reconstruction, cmap="gray")
        ax[0].axis("off")
        ax[0].title.set_text(f"Reconstructed image")

        image = get3d_middle_slice(img[0])

        ax[1].imshow(image, cmap="gray")
        ax[1].axis("off")
        ax[1].title.set_text(f"Real image")
    else:
        for i in range(n):
            reconstruction = get3d_middle_slice(reconstructions[i])

            ax[0, i].imshow(reconstruction, cmap="gray")
            ax[0, i].axis("off")
            ax[0, i].title.set_text(f"Reconstructed image {i}")

            image = get3d_middle_slice(img[i])

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
                   log_scale=False, save=True, file_path=None, title: str = None, low_fpr=0.001):
    auc = calculate_AUC(tprs, fprs)
    tpr_at_low_fpr = select_tpr_at_low_fprs(tprs, fprs, low_fpr)
    print("tpr at low fpr: ", tpr_at_low_fpr)

    if label:
        _label = label if log_scale else label + f" AUC={auc:.3f}"
    else:
        _label = f'TPR at {low_fpr:.3f} FPR = {tpr_at_low_fpr}' if log_scale else label + f'AUC = {auc:.3f}'
    plt.plot(fprs, tprs, label=_label)

    if tprs2 is not None and fprs2 is not None:
        assert label and label2, "Specify labels."
        auc = calculate_AUC(tprs2, fprs2)
        #tpr_at_low_fpr = select_tpr_at_low_fprs(tprs2, fprs2, low_fpr)
        _label2 = label2 if log_scale else label2 + f" AUC={auc:.3f}"
        plt.plot(fprs2, tprs2, label=_label2)

    if tprs3 is not None and fprs3 is not None:
        assert label3, "Specify label 3"
        auc = calculate_AUC(tprs3, fprs3)
        tpr_at_low_fpr = select_tpr_at_low_fprs(tprs3, fprs3, low_fpr)
        #tpr_at_low_fpr = "n/a" if tpr_at_low_fpr == 0.0 else tpr_at_low_fpr
        #_label3 = label3 + f" TPR(FPR={low_fpr}) = {tpr_at_low_fpr}" if log_scale else label3 + f" AUC={auc:.3f}"
        _label3 = label3 if log_scale else label3 + f" AUC={auc:.3f}"
        plt.plot(fprs3, tprs3, label=_label3)

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for random chance
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.subplots_adjust(left=0.2, bottom=0.2)  # avoid cutting off x-labels
    plt.legend(loc='lower right')

    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(1e-3, 1.0)
        plt.ylim(1e-3, 1.0)

    file_path = f"data/plots/ROC_{label}_{label2}_{label3}_log_scale{log_scale}.png" if file_path is None else file_path
    if title:
        plt.title(title)
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

def show_auc_tpr_plot(x_values, mean_auc, std_auc=None, xlabel=" ", ylabel="AUC", file_name=None):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6, 6)
    if std_auc is None:
        ax.plot(x_values, mean_auc, color="blue", marker="o")
    else:
        ax.errorbar(x_values, mean_auc, yerr=std_auc, color="blue", ecolor='#888888', marker="o")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    file_name = f"data/plots/{xlabel}_{ylabel}.png" if file_name is None else file_name
    plt.savefig(file_name)
    plt.clf()


def show_log_p_r(_adversary, title: str = None, file_path=None):
    log_p_r_members, log_p_r_nonmembers = [], []
    for it, b in enumerate(_adversary.true_memberships):
        if b == 1:
            log_p_r_members.append(_adversary.log_p_r[it])
        else:
            log_p_r_nonmembers.append(_adversary.log_p_r[it])

    plt.hist(log_p_r_members, color='blue', label='Members', alpha=0.5, bins=50)
    plt.hist(log_p_r_nonmembers, color='red', label='Non-members', alpha=0.5, bins=50)
    plt.legend()
    plt.xlabel("log p_R")
    plt.ylabel("Frequency")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    file_path = f"data/plots/log_p_r_hist.png" if file_path is None else file_path
    plt.savefig(file_path)
    plt.clf()


def plot_diffs(_adversary, title: str = None):
    diffs_members = []
    diffs_non_members = []
    diffs = np.array(_adversary.diffs)
    diffs = (diffs - diffs.min())/(diffs.max() - diffs.min())
    for it, b in enumerate(_adversary.true_memberships):
        if b == 1:
            diffs_members.append(diffs[it])
        else:
            diffs_non_members.append(diffs[it])

    plt.figure(figsize=(8, 5))
    plt.hist(diffs_members, color='blue', label='Members', alpha=0.5, bins=75)
    plt.hist(diffs_non_members, color='red', label='Non-members', alpha=0.5, bins=75)
    plt.legend()
    plt.xlabel("log p_S - log p_R")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.savefig("data/plots/loss_differences_DOMIAS.png")
    plt.clf()


def plot_loss_distributions_domias(_adversary, combine=True):
    log_p_s_members = []
    log_p_r_members = []
    log_p_s_nonmembers = []
    log_p_r_nonmembers = []
    for it, b in enumerate(_adversary.true_memberships):
        if b == 1:
            log_p_s_members.append(_adversary.log_p_s[it])
            log_p_r_members.append(_adversary.log_p_r[it])
        else:
            log_p_s_nonmembers.append(_adversary.log_p_s[it])
            log_p_r_nonmembers.append(_adversary.log_p_r[it])

    fig, ax = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(16, 9)
    ax[0, 0].hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.5, bins=50)
    ax[0, 0].hist(log_p_r_members, color='red', label='raw loss members', alpha=0.5, bins=50)
    ax[0, 0].legend(loc="upper left")

    ax[0, 1].hist(log_p_s_nonmembers, color='blue', label='syn loss non-members', alpha=0.5, bins=50)
    ax[0, 1].hist(log_p_r_nonmembers, color='red', label='raw loss non-members', alpha=0.5, bins=50)
    ax[0, 1].legend(loc="upper left")

    ax[1, 0].hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.5, bins=50)
    ax[1, 0].hist(log_p_s_nonmembers, color='red', label='syn loss non members', alpha=0.5, bins=50)
    ax[1, 0].legend(loc="upper right")

    ax[1, 1].hist(log_p_r_members, color='blue', label='raw loss members', alpha=0.5, bins=50)
    ax[1, 1].hist(log_p_r_nonmembers, color='red', label='raw loss nonmembers', alpha=0.5, bins=50)
    ax[0, 1].legend(loc="upper right")
    plt.xlabel("log p")
    plt.tight_layout()
    plt.savefig("data/plots/loss_distribution_DOMIAS.png")

    if combine:
        plt.figure(figsize=(12, 8))
        plt.hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.4, bins=85)
        plt.hist(log_p_s_nonmembers, color='tab:cyan', label='syn loss non-members', alpha=0.4, bins=85)
        plt.hist(log_p_r_members, color='tab:orange', label='raw loss members', alpha=0.4, bins=85)
        plt.hist(log_p_r_nonmembers, color='tab:red', label='raw loss non-members', alpha=0.4, bins=85)
        plt.legend()
        plt.xlabel("log p")
        plt.tight_layout()
        plt.savefig("data/plots/loss_distribution_DOMIAS_combined.png")