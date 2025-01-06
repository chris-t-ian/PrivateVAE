import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np
import os
import pandas as pd
from datetime import datetime
from p_vqvae.mia import Challenger, AdversaryDOMIAS, AdversaryZCalibratedDOMIAS, AdversaryZCalibratedDOMIAS2, \
    AdversaryLiRANSF, AdversaryLiRAClassifier, AdversaryLOGAN, MultiSeedAdversaryDOMIAS
from p_vqvae.visualise import show_roc_curve, plot_generated_images, plot_reconstructions, show_roc_curve_std, \
                               show_auc_tpr_plot
from p_vqvae.networks import train_transformer_and_vqvae
from p_vqvae.dataloader import get_train_loader, get_train_val_loader, RawDataSet
from p_vqvae.neural_spline_flow import OptimizedNSF

# don't forget to clear model outputs folder before implementing
# also check kwargs before implementation
TEST_MODE = True  # turn off for real attack

include_targets_in_reference_dataset = None  # either True, False or None (=included at random)

device = "cuda:4"
mpl.rcParams['axes.labelsize'] = 17
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
n_atlas = 955
low_fpr = 0.01
n_seeds = 2
now = datetime.now()

if TEST_MODE:
    downsample = 4
    padding = ((1, 2), (0, 0), (1, 2))
    # downsample = 1
    # padding = ((2, 2), (0, 0), (2, 2))
    n_targets = 955
    n_members = 200
    m_syn_images = n_targets // 2

else:
    downsample = 1
    padding = ((2, 2), (0, 0), (2, 2))
    n_targets = 955
    n_members = n_targets // 2 + 1
    m_syn_images = n_targets // 2 + 1

challenger_kwargs = {
    "n_c": n_members, #n_atlas // 2,  # number of raw images in dataloader of challenger
    "m_c": m_syn_images, #n_atlas // 2,  # number of synthetic images
    "n_targets": n_targets,
}
adversary_kwargs = {
    "background_knowledge": 0.0,
    "outlier_percentile": None,
}
raw_data_kwargs = {
    "root":"data/ATLAS_2",
    "cache_path":"data/cache",
    "downsample": downsample,
    "normalize": 1,
    "crop": ((8, 9), (12, 13), (0, 9)),
    "padding": padding}
vqvae_train_loader_kwargs = {
    "batch_size": 8,  #
    "augment_flag": True,
    "num_workers": 1
}
nsf_train_loader_kwargs = {
    "batch_size": 4,  #
    "augment_flag": True,
    "num_workers": 1
}
shadow_model_train_loader_kwargs = {
    "batch_size": 2,
    "augment_flag": False,
    "num_workers": 1
}
vqvae_kwargs = {
    "n_epochs": 200,
    "early_stopping_patience": 5,  # stop training after ... training steps of not improving val loss
    "val_interval": 2,
    "lr": 1e-4,  #
    "device": device,
#    "device_ids": [4,5],
    "dtype": torch.float32,
    "use_checkpointing": True,
    "commitment_cost": 0.25,
    "num_embeddings": 512,  #
    "embedding_dim": 64,  #
    "channels": (384, 512),  #
    "num_res_layers": 2,
    "num_res_channels": (384, 512),  #
    "downsample_parameters": ((2, 4, 1, 1), (2, 4, 1, 1)),
    "upsample_parameters": ((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
    "model_path": "model_outputs/mia",
}
transformer_kwargs = {
    "n_epochs": 200,
    "early_stopping_patience": 5,
    "device": device,
#    "vqvae_device": "cuda:6",
#    "device_ids": [4,5],
    "lr": 4e-4,  #
    "attn_layers_dim": 96,  #
    "attn_layers_depth": 12,  #
    "attn_layers_heads": 12,  #
    "model_path": "model_outputs/mia"
}
membership_classifier_kwargs = {
    "model_path": "model_outputs/mia",
}

# always use the same challenger:
challenger_half = Challenger(
    **challenger_kwargs,
    raw_data_kwargs=raw_data_kwargs,
    vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
    vqvae_kwargs=vqvae_kwargs,
    transformer_kwargs=transformer_kwargs,
    seed=0,
)

def log(name, content: dict, path="data/logs"):
    file = os.path.join(path, name + now.strftime("%d%m%Y_%H%M%S") + ".csv")
    df = pd.DataFrame.from_dict({"date": now.strftime("%d.%m.%Y_%H:%M:%S")})
    df.append(content)
    df.to_csv(file, mode='a', header=not os.path.exists(file))

def show_raw_image():
    dataset = RawDataSet(**raw_data_kwargs)
    train_loader = get_train_loader(dataset, 2, False, 1)
    img = next(iter(train_loader))['image'].cpu().float()
    plot_generated_images(img, 1, "data/plots/raw_image.png")

def show_synthetic_images():
    """Plot sampled images of the VQVAE + transformer as sanity check."""
    t_vqvae = challenger_half.t_vqvae
    syn = t_vqvae.create_synthetic_images(2)
    plot_generated_images(syn, 1, "data/plots/syn_image.png")

def show_nsf_samples():
    _adversary = AdversaryDOMIAS(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, background_knowledge=1.0)
    raw_samples, syn_samples = _adversary.sample_nsf(2)
    plot_generated_images(raw_samples, 2, "data/plots/NSF_raw_samples.png")
    plot_generated_images(syn_samples, 2, "data/plots/NSF_syn_samples.png")

def plot_learning_curve_vqvae_and_transformer():
    seed = 10
    dataset = RawDataSet(**raw_data_kwargs)
    train_loader, val_loader = get_train_val_loader(dataset, **vqvae_train_loader_kwargs)

    vqvae_kwargs["val_interval"] = 1
    vqvae_kwargs["n_epochs"] = 100
    vqvae_kwargs["early_stopping_patience"] = float('inf')
    transformer_kwargs["val_interval"] = 1
    transformer_kwargs["n_epochs"] = 100
    transformer_kwargs["early_stopping_patience"] = float('inf')

    t_vqvae, vqvae = train_transformer_and_vqvae(train_loader, vqvae_kwargs, transformer_kwargs,
                                                 saving_kwargs= {"seed": f"{seed}"}, val_loader=val_loader, seed=seed,
                                                 return_vqvae=True)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(12, 6)
    ax[0].plot(vqvae.epoch_recon_loss_list, color="blue", label="Training loss")
    ax[0].plot(vqvae.val_recon_epoch_loss_list, color="red", label="Validation loss")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].legend(loc="upper right")
    ax[0].title.set_text(f"First Stage")

    ax[1].plot(t_vqvae.epoch_ce_loss_list, color="blue", label="Training loss")
    ax[1].plot(t_vqvae.val_ce_epoch_loss_list, color="red", label="Validation loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend(loc="upper right")
    ax[1].title.set_text(f"Second Stage")
    print("saving figure")

    if vqvae.epoch_recon_loss_list is not None and t_vqvae.epoch_ce_loss_list is not None:
        plt.savefig("data/plots/learning_curves_vqvae.png")
        plt.clf()

def plot_learning_curves_nsf(data_type="raw"):
    if data_type == "raw":
        # seed = 69
        dataset = RawDataSet(**raw_data_kwargs)
        train_loader, val_loader = get_train_val_loader(dataset, **vqvae_train_loader_kwargs)
    elif data_type == "syn":
        raise NotImplementedError
    else:
        raise NotImplementedError

    nsf = OptimizedNSF(train_loader, val_loader)
    print("training nsf")
    nsf.train()

    train_log_density = nsf.log_density_list
    val_log_density = nsf.val_log_density_list
    train_loss = nsf.loss_list
    val_loss = nsf.val_loss_list

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(6, 6)
    ax.plot(train_log_density, color="blue", label="Training log density")
    ax.plot(val_log_density, color="red", label="Validation log density")
    ax.set_ylabel("log p")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper right")
    ax.title.set_text(f"NSF log densities")

    plt.savefig(f"data/plots/learning_curves_NSF_{data_type}.png")
    plt.clf()


def plot_loss_distributions_domias(combine=True):
    _adversary = AdversaryDOMIAS(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, background_knowledge=1.0)
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
    plt.savefig("data/plots/loss_distribution_DOMIAS.png")

    if combine:
        plt.figure(figsize=(12, 8))
        plt.hist(log_p_s_members, color='blue', label='syn loss members', alpha=0.4, bins=85)
        plt.hist(log_p_s_nonmembers, color='tab:cyan', label='syn loss non-members', alpha=0.4, bins=85)
        plt.hist(log_p_r_members, color='tab:orange', label='raw loss members', alpha=0.4, bins=85)
        plt.hist(log_p_r_nonmembers, color='tab:red', label='raw loss non-members', alpha=0.4, bins=85)
        plt.legend()
        plt.xlabel("log p")
        plt.savefig("data/plots/loss_distribution_DOMIAS_combined.png")


def plot_diffs(_adversary=None):
    if _adversary is None:
        _adversary = AdversaryDOMIAS(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, background_knowledge=1.0)
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
    plt.savefig("data/plots/loss_differences_DOMIAS.png")
    plt.clf()

def domias(adversary_knowledge=1.0, plot_roc=False, challenger=challenger_half, outlier_percentile=None):
    _adversary = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge,
                                 outlier_percentile)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"DOMIAS knowledge: {adversary_knowledge}")
        show_roc_curve(tprs, fprs, f"DOMIAS knowledge: {adversary_knowledge}", low_fprs=True)
    return tprs, fprs, _adversary.log_p_s, _adversary.log_p_r, _adversary.true_memberships


def domias_multiseed(n=n_seeds, _challenger_kwargs: dict = None, _adversary_kwargs: dict = None, _vqvae_kwargs: dict =None,
                     _transformer_kwargs: dict = None, show_roc=False, show_mia_score_hist=False):
    if _challenger_kwargs is None:
        _challenger_kwargs = challenger_kwargs
    if _adversary_kwargs is None:
        _adversary_kwargs = adversary_kwargs
    if _vqvae_kwargs is None:
        _vqvae_kwargs = vqvae_kwargs
    if _transformer_kwargs is None:
        _transformer_kwargs = transformer_kwargs

    log_p_s_list, log_p_r_list = [], []
    tprs_all, fprs_all = [], []
    auc_list, tpr_at_low_fpr = [], []
    true_memberships_list = []
    _challenger = None
    for seed in range(n):
        _challenger = Challenger(
            **_challenger_kwargs,
            raw_data_kwargs=raw_data_kwargs,
            vqvae_train_loader_kwargs=vqvae_train_loader_kwargs,
            vqvae_kwargs=_vqvae_kwargs,
            transformer_kwargs=_transformer_kwargs,
            seed=seed,
        )
        _adversary = AdversaryDOMIAS(_challenger, raw_data_kwargs, nsf_train_loader_kwargs, **_adversary_kwargs)
        tprs, fprs = _adversary.tprs, _adversary.fprs

        tprs_all.append(tprs)
        fprs_all.append(fprs)
        auc_list.append(_adversary.auc)
        log_p_s_list.extend(list(_adversary.log_p_s))
        log_p_r_list.extend(list(_adversary.log_p_r))
        true_memberships_list.extend(_adversary.true_memberships)

        # select tpr at low_fpr
        tp = tprs[np.argmax(np.array(fprs) <= low_fpr)] if np.any(np.array(fprs) <= low_fpr) else tprs[-1]
        tpr_at_low_fpr.append(tp)

    log_p_s, log_p_r = np.array(log_p_s_list), np.array(log_p_r_list)
    adversary = MultiSeedAdversaryDOMIAS(log_p_s,
                                         log_p_r,
                                         true_memberships_list,
                                         _challenger,
                                         raw_data_kwargs,
                                         nsf_train_loader_kwargs,
                                         **_adversary_kwargs,
                                         )
    #tprs_mean, fprs_mean, = adversary.tprs, adversary.fprs
    tpr_at_lowfpr_mean = np.mean(np.array(tpr_at_low_fpr))
    tpr_at_lowfpr_std = np.std(np.array(tpr_at_low_fpr))
    tprs_mean = np.mean(np.array(tprs_all), axis=0)
    fprs_mean = np.mean(np.array(fprs_all), axis=0)
    tprs_std = np.std(np.array(tprs_all), axis=0)
    auc_mean = adversary.auc
    auc_std = np.std(np.array(auc_list), axis=0)

    if show_roc:
        show_roc_curve_std(tprs=tprs_mean, fprs=fprs_mean, std1=tprs_std, label=f"DOMIAS", low_fprs=False)
        show_roc_curve_std(tprs=tprs_mean, fprs=fprs_mean, std1=tprs_std, label=f"DOMIAS", low_fprs=True)
    if show_mia_score_hist:
        plot_diffs(adversary)

    return {"tprs": tprs_all, "fprs": fprs_all, "tprs_mean": tprs_mean, "fprs_mean": fprs_mean, "tprs_std": tprs_std,
            "auc_mean": auc_mean, "auc_std": auc_std, "tpr_at_lowfpr_mean": tpr_at_lowfpr_mean,
            "tpr_at_lowfpr_std": tpr_at_lowfpr_std}

def domias_overfitting(show_roc=True, show_aucs=True):
    epoch_list = [(None, None), (100, 100), (200, 200)]
    mean_auc, std_auc = [], []
    mean_tpr_lf, std_tpr_lf = [], []
    res = []
    for epochs in epoch_list:
        if epochs[0] is None:
            vqvae_kwargs["n_epochs"] = 200
            vqvae_kwargs["early_stopping_patience"] = 5
        else:
            vqvae_kwargs["n_epochs"] = epochs[0]
            vqvae_kwargs["early_stopping_patience"] = float('inf')

        if epochs[1] is None:
            transformer_kwargs["n_epochs"] = 200
            transformer_kwargs["early_stopping_patience"] = 5
        else:
            transformer_kwargs["n_epochs"] = epochs[1]
            transformer_kwargs["early_stopping_patience"] = float('inf')

        adv_res = domias_multiseed(n_seeds,
                                   challenger_kwargs,
                                   adversary_kwargs,
                                   vqvae_kwargs,
                                   transformer_kwargs,
                                   show_roc=False,
                                   show_mia_score_hist=False)

        log_content = {**adv_res, **challenger_kwargs, **adversary_kwargs, **vqvae_kwargs,
                       **transformer_kwargs, **vqvae_train_loader_kwargs, **nsf_train_loader_kwargs}
        log("overfitting", log_content)
        res.append(adv_res)
        mean_auc.append(adv_res["auc_mean"])
        std_auc.append(adv_res["auc_std"])
        mean_tpr_lf.append(adv_res["tpr_at_lowfpr_mean"])
        std_tpr_lf.append(adv_res["tpr_at_lowfpr_std"])

    #if show_roc:
    #    show_roc_curve_std(res[0]["tprs_mean"], res[0]["fprs_mean"], f"early stopping", tprs_list[1], fprs_list[1], f"100 epochs",
    #                   tprs_list[2], fprs_list[2], f"200 epochs")
    #    show_roc_curve_std(tprs_list[0], fprs_list[0], f"early stopping", tprs_list[1], fprs_list[1], f"100 epochs",
    #                   tprs_list[2], fprs_list[2], f"200 epochs", low_fprs=True)
    if show_aucs:
        show_auc_tpr_plot([65, 100, 200], mean_auc, std_auc, xlabel="epoch", ylabel="AUC")
        show_auc_tpr_plot([65, 100, 200], mean_tpr_lf, std_tpr_lf, xlabel="epoch", ylabel=f"TPR at {low_fpr} FPR")


def domias_training_set_size():
    n_members_list = [n_targets // 8, n_targets // 4, n_targets // 2]

    for _n_members in n_members_list:
        challenger = Challenger(
            _n_members,
            m_syn_images,
            raw_data_kwargs,
            vqvae_train_loader_kwargs,
            vqvae_kwargs,
            transformer_kwargs,
        )


def domias_outlier(adversary_knowledge=1.0, plot_roc=False, challenger=challenger_half):
    _adversary_all = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge, None)
    tprs_all, fprs_all = _adversary_all.tprs, _adversary_all.fprs

    _adversary_25 = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge, .25)
    tprs_25, fprs_25 = _adversary_25.tprs, _adversary_25.fprs

    _adversary_10 = AdversaryDOMIAS(challenger, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge, .1)
    tprs_10, fprs_10 = _adversary_10.tprs, _adversary_10.fprs

    if plot_roc:
        show_roc_curve(tprs_all, fprs_all, f"All targets", tprs_25, fprs_25, f"25th percentile outliers", tprs_10,
                       fprs_10, f"10th percentile outliers")
        show_roc_curve(tprs_all, fprs_all, f"All targets", tprs_25, fprs_25, f"25th percentile outliers", tprs_10,
                       fprs_10, f"10th percentile outliers", low_fprs=True)
    return tprs_10, fprs_10

def domias_multi_seed_outlier(show_auc=True):
    outlier_percentiles = [None, .5, .25, .1]
    mean_auc, std_auc = [], []
    mean_tpr_lf, std_tpr_lf = [], []
    res = []
    for percentile in outlier_percentiles:
        adversary_kwargs["outlier_percentile"] = percentile
        challenger_kwargs["n_targets"] = int(np.ceil(outlier_percentiles[-1] * n_atlas / percentile))

        adv_res = domias_multiseed(n_seeds,
                                   challenger_kwargs,
                                   adversary_kwargs,
                                   vqvae_kwargs,
                                   transformer_kwargs,
                                   show_roc=False,
                                   show_mia_score_hist=False)

        log_content = {**adv_res, **challenger_kwargs, **adversary_kwargs, **vqvae_kwargs,
                       **transformer_kwargs, **vqvae_train_loader_kwargs, **nsf_train_loader_kwargs}
        log("outlier_selection", log_content)
        res.append(adv_res)
        mean_auc.append(adv_res["auc_mean"])
        std_auc.append(adv_res["auc_std"])
        mean_tpr_lf.append(adv_res["tpr_at_lowfpr_mean"])
        std_tpr_lf.append(adv_res["tpr_at_lowfpr_std"])

    show_roc_curve_std(tprs=res[0]["tprs_mean"], std1=res[0]["tprs_std"], fprs=res[0]["fprs_mean"], label="all targets",
                       tprs2=res[1]["tprs_mean"], std2=res[1]["tprs_std"], fprs2=res[1]["fprs_mean"], label2=f"{outlier_percentiles[1]}",
                       tprs3=res[-1]["tprs_mean"], std3=res[-1]["tprs_std"], fprs3=res[-1]["fprs_mean"], label3=f"{outlier_percentiles[-1]}",
                       )
    show_roc_curve_std(tprs=res[0]["tprs_mean"], std1=res[0]["tprs_std"], fprs=res[0]["fprs_mean"], label="all targets",
                       tprs2=res[1]["tprs_mean"], std2=res[1]["tprs_std"], fprs2=res[1]["fprs_mean"], label2=f"{outlier_percentiles[1]}",
                       tprs3=res[-1]["tprs_mean"], std3=res[-1]["tprs_std"], fprs3=res[-1]["fprs_mean"], label3=f"{outlier_percentiles[-1]}",
                       low_fprs=True
                       )

    if show_auc:
        show_auc_tpr_plot([1.0, 0.5, 0.25, 0.1], mean_auc, std_auc, xlabel="Percentile", ylabel="AUC")
        show_auc_tpr_plot([1.0, 0.5, 0.25, 0.1], mean_tpr_lf, std_tpr_lf, xlabel="Percentile", ylabel=f"TPR at {low_fpr} FPR")


def logan(adversary_knowledge=1.0, plot_roc=False):
    _adversary = AdversaryLOGAN(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs, adversary_knowledge)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"LOGAN, knowledge: {adversary_knowledge}")
        show_roc_curve(tprs, fprs, f"LOGAN, knowledge: {adversary_knowledge}", low_fprs=True)
    return tprs, fprs


def domias_with_augmentation(adversary_knowledge=0.0, plot_roc=False):
    _nsf_train_loader_kwargs = {
        "batch_size": 4,
        "augment_flag": True,
        "num_workers": 1
    }
    _adversary = AdversaryDOMIAS(challenger_half, raw_data_kwargs, _nsf_train_loader_kwargs, adversary_knowledge)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def zdomias_p_r_z(adversary_knowledge=1.0, n_z=100, plot_roc=False):
    _adversary = AdversaryZCalibratedDOMIAS(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs,
                                            adversary_knowledge, n_z)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def zdomias_p_s_z(adversary_knowledge=1.0, n_z=100, plot_roc=False):
    _adversary = AdversaryZCalibratedDOMIAS2(challenger_half, raw_data_kwargs, nsf_train_loader_kwargs,
                                             adversary_knowledge, n_z)
    tprs, fprs = _adversary.tprs, _adversary.fprs

    if plot_roc:
        show_roc_curve(tprs, fprs, f"adversary knowledge: {adversary_knowledge}")
    return tprs, fprs


def domias_vary_knowledge():
    pass
    #tprs_1, fprs_1 = multiple_challenger_seeds(n=4, adversary_knowledge=1.0)
    #tprs_2, fprs_2 = multiple_challenger_seeds(n=4, adversary_knowledge=0.5)
    #tprs_3, fprs_3 = multiple_challenger_seeds(n=4, adversary_knowledge=0.0)
    #show_roc_curve(tprs_1, fprs_1, f"100% knowledge", tprs_2, fprs_2, f"50% knowledge", tprs_3,
    #               fprs_3, f"0% knowledge", )
    #show_roc_curve(tprs_1, fprs_1, f"100% knowledge", tprs_2, fprs_2, f"50% knowledge", tprs_3,
    #               fprs_3, f"0% knowledge", low_fprs=True)


def zdomias_vs_domias():
    tprs2, fprs2 = zdomias_p_r_z(1.0, n_z=50)
    tprs1, fprs1, _, _, _ = domias(1.0)
    show_roc_curve(tprs1, fprs1, f"DOMIAS", tprs2, fprs2, f"z-DOMIAS")
    show_roc_curve(tprs1, fprs1, f"DOMIAS", tprs2, fprs2, f"z-DOMIAS", low_fprs=True)


def zdomias_vary_z():
    tprs1, fprs1 = zdomias_p_r_z(1.0, 20)
    tprs2, fprs2 = zdomias_p_r_z(1.0, 50)
    tprs3, fprs3 = zdomias_p_r_z(1.0, 100)
    show_roc_curve(tprs1, fprs1, f"z-DOMIAS, n_Z=20", tprs2, fprs2, f"z-DOMIAS, n_Z=50", tprs3, fprs3,
                   f"z-DOMIAS, n_Z=100")
    show_roc_curve(tprs1, fprs1, f"z-DOMIAS, n_Z=20", tprs2, fprs2, f"z-DOMIAS, n_Z=50", tprs3, fprs3,
                   f"z-DOMIAS, n_Z=100", low_fprs=True)


def zdomias_vary_knowledge():
    tprs_1, fprs_1 = zdomias_p_r_z(1.0)
    tprs_2, fprs_2 = zdomias_p_r_z(0.5)
    tprs_3, fprs_3 = zdomias_p_r_z(0.0)
    show_roc_curve(tprs_1, fprs_1, f"100% adversary knowledge", tprs_2, fprs_2, f"50% adversary knowledge", tprs_3,
                   fprs_3, f"0% adversary knowledge")
    show_roc_curve(tprs_1, fprs_1, f"100% adversary knowledge", tprs_2, fprs_2, f"50% adversary knowledge", tprs_3,
                   fprs_3, f"0% adversary knowledge", low_fprs=True)


def zdomias_repeat():
    tprs1, fprs1 = zdomias_p_r_z(1.0, 455)
    tprs2, fprs2 = zdomias_p_r_z(1.0, 455)
    tprs3, fprs3 = zdomias_p_r_z(1.0, 455)
    show_roc_curve(tprs1, fprs1, f"z-DOMIAS1", tprs2, fprs2, f"z-DOMIAS2", tprs3, fprs3, f"z-DOMIAS3", )


def zdomias_p_r_z_vs_p_s_z():
    tprs1, fprs1 = zdomias_p_r_z(1.0, 455)
    tprs2, fprs2 = zdomias_p_s_z(1.0, 455)
    tprs3, fprs3 = zdomias_p_s_z(1.0, 455)
    show_roc_curve(tprs1, fprs1, f"p_r_z-DOMIAS", tprs2, fprs2, f"p_s_z-DOMIAS1", tprs3, fprs3, f"p_s_z-DOMIAS2", )


def zdomias_p_s_z_repeat():
    tprs1, fprs1 = zdomias_p_s_z(1.0, 455)
    tprs2, fprs2 = zdomias_p_s_z(1.0, 455)
    tprs3, fprs3 = zdomias_p_s_z(1.0, 455)
    show_roc_curve(tprs1, fprs1, f"p_s_z-DOMIAS1", tprs2, fprs2, f"p_s_z-DOMIAS2", tprs3, fprs3, f"p_s_z-DOMIAS3", )


def LIRA_NSF(plot_roc=True):
    challenger = Challenger(
        n_members,
        m_syn_images,
        raw_data_kwargs,
        vqvae_train_loader_kwargs,
        vqvae_kwargs,
        transformer_kwargs,
        n_targets=100
    )
    _adversary = AdversaryLiRANSF(
        challenger,
        offline=False,
        n_reference_models=1,
        background_knowledge=1.0,
        shadow_model_train_loader_kwargs=shadow_model_train_loader_kwargs,
    )

    tprs, fprs = _adversary.tprs, _adversary.fprs
    if plot_roc:
        show_roc_curve(tprs, fprs, f"LiRA NSF adversary knowledge: {1.0}")

    return tprs, fprs


if __name__ == "__main__":
    show_synthetic_images()
    plot_loss_distributions_domias(combine=True)
    plot_diffs()
    domias(1.0, plot_roc=True)
    domias_vary_knowledge()
    zdomias_vs_domias()
